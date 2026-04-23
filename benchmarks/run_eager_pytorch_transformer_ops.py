"""
benchmark_attention.py — Benchmark transformer attention operations.
Measures QKV projection, dot product, softmax, and full attention
against PyTorch eager baseline.

Usage: modal run benchmark_attention.py
"""

import modal

app = modal.App("attention-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch==2.1.0",
        "numpy==1.26.4",
        "tabulate",
    ])
)

# ── Shapes to test ───────────────────────────────────────────────────────────
# seq_len, hidden_dim, num_heads, model
ATTENTION_SHAPES = [
    (128,  768,  12, "BERT"),    # BERT-base
    (512,  768,  12, "BERT"),    # BERT-base, longer seq
    (128,  4096, 32, "Llama"),   # Llama-3 8B
    (512,  4096, 32, "Llama"),   # Llama-3 8B
    (2048, 4096, 32, "Llama"),   # Llama-3 8B, long seq
]


def timeit(fn, iters=100, warmup=20):
    """Time a function on GPU, returns milliseconds."""
    import torch, time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


@app.function(gpu="A100", image=image, timeout=3600)
def benchmark_attention(seq_len: int, hidden_dim: int, num_heads: int, model: str):
    import torch
    import torch.nn.functional as F
    import math

    print(f"\n{'='*60}")
    print(f"Model: {model} | seq_len={seq_len} | hidden={hidden_dim} | heads={num_heads}")
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    head_dim = hidden_dim // num_heads  # dimension per attention head
    dtype = torch.float16

    # ── Input tensors ────────────────────────────────────────────────────────
    # X is the input sequence: (seq_len, hidden_dim)
    X  = torch.randn(seq_len, hidden_dim, dtype=dtype).cuda()

    # Weight matrices for Q, K, V projections
    Wq = torch.randn(hidden_dim, hidden_dim, dtype=dtype).cuda()
    Wk = torch.randn(hidden_dim, hidden_dim, dtype=dtype).cuda()
    Wv = torch.randn(hidden_dim, hidden_dim, dtype=dtype).cuda()

    results = {
        "model": model,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
    }

    # ── 1. QKV Projections ───────────────────────────────────────────────────
    # Three separate GEMMs: Q = X @ Wq, K = X @ Wk, V = X @ Wv
    # This is the most compute-heavy part
    def qkv_separate():
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        return Q, K, V

    ms = timeit(qkv_separate)
    results["qkv_ms"] = ms
    print(f"  QKV projections (3 separate GEMMs): {ms:.3f} ms")

    # ── 2. Fused QKV (single GEMM) ───────────────────────────────────────────
    # Stack all three weight matrices into one and do a single bigger GEMM
    # This is faster because one big GEMM is more efficient than 3 small ones
    W_qkv = torch.randn(hidden_dim, hidden_dim * 3, dtype=dtype).cuda()

    def qkv_fused():
        QKV = X @ W_qkv  # one big GEMM instead of three
        Q, K, V = QKV.split(hidden_dim, dim=-1)
        return Q, K, V

    ms = timeit(qkv_fused)
    results["qkv_fused_ms"] = ms
    print(f"  QKV projections (fused single GEMM): {ms:.3f} ms")

    # Precompute Q, K, V for the remaining benchmarks
    Q, K, V = qkv_separate()

    # Reshape for multi-head attention:
    # (seq_len, hidden_dim) → (num_heads, seq_len, head_dim)
    Q = Q.view(seq_len, num_heads, head_dim).transpose(0, 1)
    K = K.view(seq_len, num_heads, head_dim).transpose(0, 1)
    V = V.view(seq_len, num_heads, head_dim).transpose(0, 1)

    # ── 3. Dot Product (Q @ K^T) ─────────────────────────────────────────────
    # How much each query attends to each key
    # Shape: (num_heads, seq_len, seq_len)
    def dot_product():
        return torch.bmm(Q, K.transpose(-2, -1))

    ms = timeit(dot_product)
    results["dot_product_ms"] = ms
    print(f"  Dot product (Q @ K^T):               {ms:.3f} ms")

    scores = dot_product()

    # ── 4. Scale + Softmax ───────────────────────────────────────────────────
    # Divide by sqrt(head_dim) to stabilize gradients, then softmax
    scale = math.sqrt(head_dim)

    def scale_softmax():
        return F.softmax(scores / scale, dim=-1)

    ms = timeit(scale_softmax)
    results["softmax_ms"] = ms
    print(f"  Scale + Softmax:                     {ms:.3f} ms")

    attn_weights = scale_softmax()

    # ── 5. Weighted Sum (scores @ V) ─────────────────────────────────────────
    # Use attention weights to mix the values
    def weighted_sum():
        return torch.bmm(attn_weights, V)

    ms = timeit(weighted_sum)
    results["weighted_sum_ms"] = ms
    print(f"  Weighted sum (attn @ V):             {ms:.3f} ms")

    # ── 6. Full Attention (all steps together) ───────────────────────────────
    # This is what PyTorch actually does when you call it end to end
    def full_attention():
        Q_ = (X @ Wq).view(seq_len, num_heads, head_dim).transpose(0, 1)
        K_ = (X @ Wk).view(seq_len, num_heads, head_dim).transpose(0, 1)
        V_ = (X @ Wv).view(seq_len, num_heads, head_dim).transpose(0, 1)
        scores_ = torch.bmm(Q_, K_.transpose(-2, -1)) / scale
        attn_ = F.softmax(scores_, dim=-1)
        return torch.bmm(attn_, V_)

    ms = timeit(full_attention)
    results["full_attention_ms"] = ms
    print(f"  Full attention (all steps):          {ms:.3f} ms")

    # ── 7. PyTorch optimized attention (F.scaled_dot_product_attention) ──────
    # PyTorch's built-in fused attention — uses FlashAttention under the hood
    # This is the gold standard to compare against
    def pytorch_sdpa():
        Q_ = (X @ Wq).view(seq_len, num_heads, head_dim).transpose(0, 1)
        K_ = (X @ Wk).view(seq_len, num_heads, head_dim).transpose(0, 1)
        V_ = (X @ Wv).view(seq_len, num_heads, head_dim).transpose(0, 1)
        return F.scaled_dot_product_attention(Q_, K_, V_)

    ms = timeit(pytorch_sdpa)
    results["sdpa_ms"] = ms
    print(f"  PyTorch SDPA (FlashAttention):       {ms:.3f} ms")

    # ── Memory usage ─────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    full_attention()
    torch.cuda.synchronize()
    results["memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  Peak memory (full attention):        {results['memory_mb']:.1f} MB")

    return results


@app.local_entrypoint()
def main():
    from tabulate import tabulate

    print("="*70)
    print("Transformer Attention Benchmark — PyTorch Eager")
    print("="*70)

    # Run all shapes
    all_results = list(benchmark_attention.starmap(ATTENTION_SHAPES))

    # ── Per-operation breakdown ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("PER-OPERATION BREAKDOWN (ms)")
    print("="*70)

    op_rows = []
    for r in all_results:
        op_rows.append([
            r["model"],
            r["seq_len"],
            f"{r['qkv_ms']:.3f}",
            f"{r['qkv_fused_ms']:.3f}",
            f"{r['dot_product_ms']:.3f}",
            f"{r['softmax_ms']:.3f}",
            f"{r['weighted_sum_ms']:.3f}",
            f"{r['full_attention_ms']:.3f}",
            f"{r['sdpa_ms']:.3f}",
            f"{r['memory_mb']:.1f}",
        ])

    op_headers = [
        "Model", "Seq Len",
        "QKV (3x)", "QKV Fused",
        "Q@K^T", "Softmax", "Attn@V",
        "Full Attn", "SDPA",
        "Mem (MB)",
    ]
    print(tabulate(op_rows, headers=op_headers, tablefmt="rounded_outline"))

    # ── Fusion benefit ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FUSION BENEFIT: Separate ops vs PyTorch SDPA")
    print("="*70)

    fusion_rows = []
    for r in all_results:
        separate_total = r["qkv_ms"] + r["dot_product_ms"] + r["softmax_ms"] + r["weighted_sum_ms"]
        speedup = separate_total / r["sdpa_ms"]
        fusion_rows.append([
            r["model"],
            r["seq_len"],
            f"{separate_total:.3f}",
            f"{r['sdpa_ms']:.3f}",
            f"{speedup:.2f}x",
        ])

    fusion_headers = ["Model", "Seq Len", "Separate Ops (ms)", "SDPA (ms)", "SDPA Speedup"]
    print(tabulate(fusion_rows, headers=fusion_headers, tablefmt="rounded_outline"))
    print("\nSDPA speedup > 1x = fused attention is faster than running ops separately")

    # Save results
    import json
    with open("attention_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nRaw results saved to attention_results.json")