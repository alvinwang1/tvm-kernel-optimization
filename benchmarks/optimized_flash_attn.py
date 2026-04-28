import modal

try:
    from modal_app import image
except ImportError:
    image = None

app = modal.App("optimized-attention")

import torch
import time

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is None:
    def jit(f): return f
    class Mock:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    tl = Mock()
else:
    jit = triton.jit

# ── TRITON KERNEL ───────────────────────────────────────────────────────────

@jit
def _flash_fwd_kernel(
    Q, K, V, sm_scale, L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Offsets
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize pointers
    q_ptr = Q + off_hz * stride_qh + start_m * BLOCK_M * stride_qm
    k_ptr = K + off_hz * stride_kh
    v_ptr = V + off_hz * stride_vh
    
    # Correct pointers for the block
    off_d = tl.arange(0, BLOCK_DMODEL)
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q (once per block_m)
    # Using tl.load with mask and padding if necessary
    q = tl.load(q_ptr + off_m[:, None] * stride_qm + off_d[None, :] * stride_qk)
    
    # Loop over K, V in blocks of BLOCK_N
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V block
        k = tl.load(k_ptr + (start_n + off_n)[None, :] * stride_kn + off_d[:, None] * stride_kk)
        v = tl.load(v_ptr + (start_n + off_n)[:, None] * stride_vk + off_d[None, :] * stride_vn)
        
        # Dot product: S = (Q @ K^T) * scale
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        # Online Softmax updates
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Update accumulator
        acc *= alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        
        # Update normalization factors
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]
    
    # Write output
    out_ptr = Out + off_hz * stride_oh + start_m * BLOCK_M * stride_om
    tl.store(out_ptr + off_m[:, None] * stride_om + off_d[None, :] * stride_on, acc.to(tl.float16))
    
    # Store LSE for backward if needed
    l_ptr = L + off_hz * N_CTX + start_m * BLOCK_M
    # tl.store(l_ptr + off_m, m_i + tl.log(l_i))

# ── WRAPPER ──────────────────────────────────────────────────────────────────

def flash_attention_sm89(q, k, v, causal=False, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    L = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    
    # Tuned for Ada (L40S) to fit within 100KB SMEM limit
    BLOCK_M = 128
    BLOCK_N = 64
    num_stages = 2  # Reduced stages to allow larger BLOCK_M
    num_warps = 8
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    
    _flash_fwd_kernel[grid](
        q, k, v, sm_scale, L, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
        GROUP_SIZE_M=8,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

# ── BENCHMARK ───────────────────────────────────────────────────────────────

@app.function(image=image, gpu="L40S")
def benchmark():
    import torch.nn.functional as F
    B, H, S, D = 1, 32, 2048, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(20):
        flash_attention_sm89(q, k, v)
        F.scaled_dot_product_attention(q, k, v)
    
    iters = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        res_triton = flash_attention_sm89(q, k, v)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - t0) / iters * 1000
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        res_pt = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / iters * 1000
    
    # Accuracy check
    diff = (res_triton - res_pt).abs().max()
    
    print(f"\nWorkload: B={B}, H={H}, S={S}, D={D}")
    print(f"PyTorch SDPA (FA2/Fused): {pt_ms:.4f} ms")
    print(f"Custom Triton (sm_89 Opt): {triton_ms:.4f} ms")
    print(f"Speedup: {pt_ms / triton_ms:.2f}x")
    print(f"Max Difference: {diff:.6e}")
    
    if diff < 1e-2:
        print("Accuracy: VALID ✓")
    else:
        print("Accuracy: INVALID ✗ (Difference too high)")

@app.local_entrypoint()
def main():
    benchmark.remote()

if __name__ == "__main__":
    from torch.nn import functional as F
    # benchmark()
