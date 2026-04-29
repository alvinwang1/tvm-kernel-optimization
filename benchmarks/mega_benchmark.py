import modal
import os
import sys
from pathlib import Path

app = modal.App("tvm-mega-benchmark-v2")
volume = modal.Volume.from_name("tvm-tuning-results")

# Architecture configuration
GPU_TYPE = os.environ.get("MODAL_GPU", "L40S")

try:
    from modal_app import image
except ImportError:
    image = None

# ------------------------------------------------------------------------------
# Benchmark Suite (Everything inside to avoid local import errors)
# ------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU_TYPE, volumes={"/tuning_results": volume}, timeout=3600)
def mega_benchmark():
    import torch
    import torch.nn.functional as F
    import time
    import numpy as np
    import triton
    import triton.language as tl
    import tvm
    from tvm import meta_schedule as ms
    import tvm.te as te
    import tvm.contrib.nvcc

    # --- Triton Kernel ---
    @triton.jit
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
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        q_ptr = Q + off_hz * stride_qh + start_m * BLOCK_M * stride_qm
        k_ptr = K + off_hz * stride_kh
        v_ptr = V + off_hz * stride_vh
        off_d = tl.arange(0, BLOCK_DMODEL)
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        q = tl.load(q_ptr + off_m[:, None] * stride_qm + off_d[None, :] * stride_qk)
        for start_n in range(0, N_CTX, BLOCK_N):
            k = tl.load(k_ptr + (start_n + off_n)[None, :] * stride_kn + off_d[:, None] * stride_kk)
            v = tl.load(v_ptr + (start_n + off_n)[:, None] * stride_vk + off_d[None, :] * stride_vn)
            qk = tl.dot(q, k) * sm_scale
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])
            acc *= alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
        acc = acc / l_i[:, None]
        out_ptr = Out + off_hz * stride_oh + start_m * BLOCK_M * stride_om
        tl.store(out_ptr + off_m[:, None] * stride_om + off_d[None, :] * stride_on, acc.to(tl.float16))

    def triton_flash_attn(q, k, v, causal=False, sm_scale=None):
        if sm_scale is None: sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        batch, heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)
        L = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N = 128, 64
        num_stages, num_warps = 2, 8
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
        _flash_fwd_kernel[grid](
            q, k, v, sm_scale, L, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, heads, seq_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
            GROUP_SIZE_M=8, num_stages=num_stages, num_warps=num_warps,
        )
        return out

    # --- TVM Loader ---
    def load_tvm_fused_kernel(S, H, D, arch):
        try: from tvm.tir.tensor_intrin import cuda as _
        except: pass
        def make_qk_mod(S, H, D, inv_sc):
            Q, K = te.placeholder((H, S, D), name="Q", dtype="float16"), te.placeholder((H, S, D), name="K", dtype="float16")
            k1, inv_c = te.reduce_axis((0, D), "k1"), tvm.tir.const(inv_sc, "float16")
            QK = te.compute((H, S, S), lambda h, i, j: te.sum(Q[h, i, k1] * K[h, j, k1] * inv_c, axis=k1), name="QK")
            return tvm.IRModule({"main": te.create_prim_func([Q, K, QK])})
        def make_softmax_av_mod(S, H, D):
            QK, V = te.placeholder((H, S, S), name="QK", dtype="float16"), te.placeholder((H, S, D), name="V", dtype="float16")
            m_ax = te.reduce_axis((0, S), "m_ax")
            RowMax = te.compute((H, S), lambda h, i: te.max(QK[h, i, m_ax], axis=m_ax), name="RowMax")
            Exp = te.compute((H, S, S), lambda h, i, j: tvm.te.exp(QK[h, i, j] - RowMax[h, i]), name="Exp")
            s_ax = te.reduce_axis((0, S), "s_ax")
            RowSum = te.compute((H, S), lambda h, i: te.sum(Exp[h, i, s_ax], axis=s_ax), name="RowSum")
            Attn = te.compute((H, S, S), lambda h, i, j: Exp[h, i, j] / RowSum[h, i], name="Attn")
            k2 = te.reduce_axis((0, S), "k2")
            Out = te.compute((H, S, D), lambda h, i, d: te.sum(Attn[h, i, k2] * V[h, k2, d], axis=k2), name="Out")
            return tvm.IRModule({"main": te.create_prim_func([QK, V, Out])})
        qk_tag, sav_tag = f"qk_gemm_v6_{S}_{H}_{D}_{arch}", f"softmax_av_v6_{S}_{H}_{D}_{arch}"
        target = tvm.target.Target(f"cuda -arch={arch}")
        def load_db_kernel(mod, work_dir):
            if not os.path.exists(work_dir): return None
            db = ms.database.JSONDatabase(work_dir=work_dir)
            records = db.get_all_tuning_records()
            if not records: return None
            best_record = sorted(records, key=lambda r: sum(r.run_secs)/len(r.run_secs) if r.run_secs else float('inf'))[0]
            sch = tvm.tir.Schedule(mod)
            best_record.trace.apply_to_schedule(sch, remove_postproc=False)
            with tvm.transform.PassContext(opt_level=3): return tvm.build(sch.mod, target=target)
        return load_db_kernel(make_qk_mod(S, H, D, 1.0/(D**0.5)), f"/tuning_results/attn_v6_{qk_tag}"), \
               load_db_kernel(make_softmax_av_mod(S, H, D), f"/tuning_results/attn_v6_{sav_tag}")

    # --- Run Benchmark ---
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    print(f"Running Mega-Benchmark on {torch.cuda.get_device_name(0)} ({arch})")
    WORKLOADS = [
        (128, 12, 64, "BERT-128"),
        (512, 12, 64, "BERT-512"),
        (128, 32, 128, "Llama-128"),
        (512, 32, 128, "Llama-512"),
        (2048, 32, 128, "Llama-2048")
    ]
    results = []
    for S, H, D, name in WORKLOADS:
        print(f"\n>>> Workload: {name} (S={S}, H={H}, D={D})")
        q, k, v = [torch.randn(1, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3)]
        def run_sdpa(): return F.scaled_dot_product_attention(q, k, v)
        for _ in range(20): run_sdpa()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(200): run_sdpa()
        torch.cuda.synchronize(); sdpa_ms = (time.perf_counter() - t0) / 200 * 1000
        def run_triton(): return triton_flash_attn(q, k, v)
        try:
            for _ in range(20): run_triton()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(200): run_triton()
            torch.cuda.synchronize(); triton_ms = (time.perf_counter() - t0) / 200 * 1000
        except Exception as e: print(f"  [Triton] Error: {e}"); triton_ms = float('nan')
        tvm_ms = float('nan')
        try:
            lib_qk, lib_sav = load_tvm_fused_kernel(S, H, D, arch)
            if lib_qk and lib_sav:
                qk_buf = torch.empty((1, H, S, S), device="cuda", dtype=torch.float16)
                out_buf = torch.empty((1, H, S, D), device="cuda", dtype=torch.float16)
                def run_tvm():
                    lib_qk(tvm.nd.from_dlpack(q.squeeze(0)), tvm.nd.from_dlpack(k.squeeze(0)), tvm.nd.from_dlpack(qk_buf.squeeze(0)))
                    lib_sav(tvm.nd.from_dlpack(qk_buf.squeeze(0)), tvm.nd.from_dlpack(v.squeeze(0)), tvm.nd.from_dlpack(out_buf.squeeze(0)))
                for _ in range(20): run_tvm()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(200): run_tvm()
                torch.cuda.synchronize(); tvm_ms = (time.perf_counter() - t0) / 200 * 1000
        except Exception as e: print(f"  [TVM] Error: {e}")
        results.append({"name": name, "sdpa_ms": sdpa_ms, "triton_ms": triton_ms, "tvm_ms": tvm_ms})

    print("\n" + "="*60 + f"\nMEGA-BENCHMARK SUMMARY ({arch})\n" + "="*60)
    for r in results:
        print(f"{r['name']}:")
        print(f"  PyTorch SDPA:  {r['sdpa_ms']:.4f} ms")
        print(f"  Hand-Tuned Tri:{r['triton_ms']:.4f} ms ({r['sdpa_ms']/r['triton_ms']:.2f}x vs SDPA)")
        print(f"  TVM Fused v6:  {r['tvm_ms']:.4f} ms ({r['sdpa_ms']/r['tvm_ms']:.2f}x vs SDPA)")
    return results

@app.local_entrypoint()
def main(): mega_benchmark.remote()
