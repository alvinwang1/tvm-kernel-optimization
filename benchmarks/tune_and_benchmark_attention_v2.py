"""
tune_and_benchmark_attention_v2.py
===================================
Fairness-focused benchmark for TVM vs PyTorch attention components.

What this v2 adds:
1. Keeps the original script untouched.
2. Reports both:
   - PyTorch batched path (best practical path)
   - PyTorch per-head loop path (matched to current TVM execution style)
3. Compares TVM against the matched PyTorch loop path for QK/AV.
4. Separates kernel-pipeline comparison (QKV + QK + AV) from SDPA fused path.

Usage:
    modal run tune_and_benchmark_attention_v2.py
    modal run tune_and_benchmark_attention_v2.py --max-trials 200 --iters 100
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import modal

try:
    from app import image
except ModuleNotFoundError:
    image = None

app = modal.App("tvm-attention-benchmark-v2")

volume = modal.Volume.from_name("tvm-tuning-results", create_if_missing=True)

# (seq_len, hidden_dim, num_heads, model_name)
ATTENTION_SHAPES = [
    (128, 768, 12, "BERT"),
    (512, 768, 12, "BERT"),
    (128, 4096, 32, "Llama"),
    (512, 4096, 32, "Llama"),
    (2048, 4096, 32, "Llama"),
]


def _timeit(fn, iters=200, warmup=20):
    import time
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


@app.function(
    image=image,
    gpu="L40S",
    timeout=86400,
    volumes={"/tuning_results": volume},
)
def tune_and_benchmark_attention_v2(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    model: str,
    max_trials: int = 800,
    iters: int = 200,
    pytorch_only: bool = False,
):
    import math
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tvm
    import tvm.te as te
    from tvm import meta_schedule as ms

    assert torch.cuda.is_available(), "No CUDA device found"

    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    dev = tvm.cuda(0)

    try:
        target = tvm.target.Target.from_device(dev)
    except Exception:
        target = tvm.target.Target(
            {
                "kind": "cuda",
                "arch": arch,
                "max_num_threads": 1024,
                "max_threads_per_block": 1024,
                "max_shared_memory_per_block": 49152,
                "thread_warp_size": 32,
            }
        )

    head_dim = hidden_dim // num_heads
    scale = math.sqrt(head_dim)

    print(f"\n{'=' * 68}")
    print(f"[v2] {model} seq={seq_len} hidden={hidden_dim} heads={num_heads} on {gpu_name} ({arch})")
    print(f"{'=' * 68}")

    # Inputs
    X = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device="cuda")
    Wq = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")
    Wk = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")
    Wv = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")

    def make_gemm_mod(M, N, K):
        k_ax = te.reduce_axis((0, K), name="k")
        A = te.placeholder((M, K), name="A", dtype="float16")
        B = te.placeholder((K, N), name="B", dtype="float16")
        C = te.compute((M, N), lambda i, j: te.sum(A[i, k_ax] * B[k_ax, j], axis=k_ax), name="C")
        func = te.create_prim_func([A, B, C])
        return tvm.IRModule({"main": func}), A, B, C

    def prepare_gemm(M, N, K, tag):
        work_dir = f"/tuning_results/attn_v2_{tag}_{M}_{N}_{K}_{arch}"
        mod, A_ph, B_ph, C_ph = make_gemm_mod(M, N, K)

        tuning_record = f"{work_dir}/database_tuning_record.json"
        workload_record = f"{work_dir}/database_workload.json"
        has_cached_db = (
            os.path.exists(tuning_record)
            and os.path.exists(workload_record)
            and os.path.getsize(tuning_record) > 0
            and os.path.getsize(workload_record) > 0
        )

        def build_from_database():
            try:
                database = ms.database.JSONDatabase(work_dir=work_dir)
                sch = ms.tir_integration.compile_tir(database, mod, target)
                if sch is None:
                    return None
                with tvm.transform.PassContext(opt_level=3):
                    return tvm.build(sch.mod, target=target)
            except Exception as err:
                print(f"  [cache-invalid] {tag} ({M},{N},{K}) {type(err).__name__}: {err}")
                return None

        if has_cached_db:
            print(f"  [cache] {tag} ({M},{N},{K})")
            lib = build_from_database()
            if lib is not None:
                return lib, True
            print(f"  [retune] {tag} cache exists but could not compile")

        should_tune = (not has_cached_db) or (max_trials > 0)
        if should_tune:
            if max_trials <= 0:
                print(f"  [no-cache] {tag} ({M},{N},{K}) has no usable DB and max_trials=0")
            else:
                print(f"  [tune] {tag} ({M},{N},{K}) up to {max_trials} trials")
                ms.tune_tir(
                    mod=mod,
                    target=target,
                    max_trials_global=max_trials,
                    num_trials_per_iter=32,
                    work_dir=work_dir,
                )
                volume.commit()

                lib = build_from_database()
                if lib is not None:
                    return lib, True

        print(f"  [fallback] {tag} uses simple tiled schedule")
        s = te.create_schedule(C_ph.op)
        io, ii = s[C_ph].split(s[C_ph].op.axis[0], factor=16)
        jo, ji = s[C_ph].split(s[C_ph].op.axis[1], factor=16)
        s[C_ph].bind(io, te.thread_axis("blockIdx.y"))
        s[C_ph].bind(jo, te.thread_axis("blockIdx.x"))
        s[C_ph].bind(ii, te.thread_axis("threadIdx.y"))
        s[C_ph].bind(ji, te.thread_axis("threadIdx.x"))
        with tvm.transform.PassContext(opt_level=3):
            return tvm.build(s, [A_ph, B_ph, C_ph], target=target), False

    def to_tvm(t: torch.Tensor):
        return tvm.nd.array(t.detach().cpu().numpy(), dev)

    def zeros_tvm(shape):
        return tvm.nd.array(np.zeros(shape, dtype="float16"), dev)

    q_lib = k_lib = v_lib = qk_lib = av_lib = None
    proj_tuned = qk_tuned = av_tuned = False
    if not pytorch_only:
        # Q/K/V projections share the same shape — tune once, reuse for all three.
        proj_lib, proj_tuned = prepare_gemm(seq_len, hidden_dim, hidden_dim, "proj")
        q_lib = k_lib = v_lib = proj_lib
        qk_lib, qk_tuned = prepare_gemm(seq_len, seq_len, head_dim, "QK_dot")
        av_lib, av_tuned = prepare_gemm(seq_len, head_dim, seq_len, "AV_sum")

    results = {
        "model": model,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "gpu": gpu_name,
        "arch": arch,
        "pytorch_only": pytorch_only,
        "tuned": all([proj_tuned, qk_tuned, av_tuned]) if not pytorch_only else None,
    }

    print("\n[PyTorch baselines]")

    def pt_qkv():
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        return Q, K, V

    results["pt_qkv_ms"] = _timeit(pt_qkv, iters=iters)

    Q_pt, K_pt, V_pt = pt_qkv()
    Q_mh = Q_pt.view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()  # (H,S,D)
    K_mh = K_pt.view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()  # (H,S,D)
    V_mh = V_pt.view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()  # (H,S,D)

    def pt_dot_batched():
        return torch.bmm(Q_mh, K_mh.transpose(-2, -1))

    def pt_dot_loop():
        outs = []
        for h in range(num_heads):
            outs.append(Q_mh[h] @ K_mh[h].transpose(-2, -1))
        return outs

    results["pt_dot_batched_ms"] = _timeit(pt_dot_batched, iters=iters)
    results["pt_dot_loop_ms"] = _timeit(pt_dot_loop, iters=iters)

    scores_batched = pt_dot_batched()

    def pt_softmax_batched():
        return F.softmax(scores_batched / scale, dim=-1)

    results["pt_softmax_ms"] = _timeit(pt_softmax_batched, iters=iters)
    attn_batched = pt_softmax_batched()

    def pt_av_batched():
        return torch.bmm(attn_batched, V_mh)

    def pt_av_loop():
        outs = []
        for h in range(num_heads):
            outs.append(attn_batched[h] @ V_mh[h])
        return outs

    results["pt_av_batched_ms"] = _timeit(pt_av_batched, iters=iters)
    results["pt_av_loop_ms"] = _timeit(pt_av_loop, iters=iters)

    def pt_full_unfused_batched():
        Q_ = (X @ Wq).view(seq_len, num_heads, head_dim).transpose(0, 1)
        K_ = (X @ Wk).view(seq_len, num_heads, head_dim).transpose(0, 1)
        V_ = (X @ Wv).view(seq_len, num_heads, head_dim).transpose(0, 1)
        S_ = torch.bmm(Q_, K_.transpose(-2, -1)) / scale
        A_ = F.softmax(S_, dim=-1)
        return torch.bmm(A_, V_)

    def pt_full_unfused_loop():
        Q_ = (X @ Wq).view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()
        K_ = (X @ Wk).view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()
        V_ = (X @ Wv).view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()
        outs = []
        for h in range(num_heads):
            s = (Q_[h] @ K_[h].transpose(-2, -1)) / scale
            a = F.softmax(s, dim=-1)
            outs.append(a @ V_[h])
        return outs

    def pt_sdpa():
        # Use explicit [B, H, S, D] contiguous layout to improve fast-path dispatch.
        Q_ = (X @ Wq).view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        K_ = (X @ Wk).view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        V_ = (X @ Wv).view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        return F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)

    results["pt_full_unfused_batched_ms"] = _timeit(pt_full_unfused_batched, iters=iters)
    results["pt_full_unfused_loop_ms"] = _timeit(pt_full_unfused_loop, iters=iters)
    results["pt_sdpa_ms"] = _timeit(pt_sdpa, iters=iters)

    # Always compute PT-only pipeline summaries.
    results["pt_pipeline_loop_no_softmax_ms"] = (
        results["pt_qkv_ms"] + results["pt_dot_loop_ms"] + results["pt_av_loop_ms"]
    )
    results["pt_pipeline_loop_with_softmax_ms"] = (
        results["pt_pipeline_loop_no_softmax_ms"] + results["pt_softmax_ms"]
    )

    if pytorch_only:
        print("[skip] TVM kernels (pytorch_only=True)")
        return results

    print("[TVM kernels]")

    X_tvm = to_tvm(X)
    Wq_tvm = to_tvm(Wq)
    Wk_tvm = to_tvm(Wk)
    Wv_tvm = to_tvm(Wv)
    Q_tvm = zeros_tvm((seq_len, hidden_dim))
    K_tvm = zeros_tvm((seq_len, hidden_dim))
    V_tvm = zeros_tvm((seq_len, hidden_dim))

    def tvm_qkv():
        q_lib(X_tvm, Wq_tvm, Q_tvm)
        k_lib(X_tvm, Wk_tvm, K_tvm)
        v_lib(X_tvm, Wv_tvm, V_tvm)
        dev.sync()

    results["tvm_qkv_ms"] = _timeit(tvm_qkv, iters=iters)

    # Build matched per-head inputs once (outside timed regions)
    Q_heads_tvm = [to_tvm(Q_mh[h]) for h in range(num_heads)]
    Kt_heads_tvm = [to_tvm(K_mh[h].transpose(-2, -1).contiguous()) for h in range(num_heads)]
    QK_outs_tvm = [zeros_tvm((seq_len, seq_len)) for _ in range(num_heads)]

    AV_in_tvm = [to_tvm(attn_batched[h]) for h in range(num_heads)]
    V_heads_tvm = [to_tvm(V_mh[h]) for h in range(num_heads)]
    AV_outs_tvm = [zeros_tvm((seq_len, head_dim)) for _ in range(num_heads)]

    def tvm_dot_loop():
        for h in range(num_heads):
            qk_lib(Q_heads_tvm[h], Kt_heads_tvm[h], QK_outs_tvm[h])
        dev.sync()

    def tvm_av_loop():
        for h in range(num_heads):
            av_lib(AV_in_tvm[h], V_heads_tvm[h], AV_outs_tvm[h])
        dev.sync()

    results["tvm_dot_loop_ms"] = _timeit(tvm_dot_loop, iters=iters)
    results["tvm_av_loop_ms"] = _timeit(tvm_av_loop, iters=iters)

    # Fair, matched pipeline (no softmax): same granularity on both sides.
    results["pt_pipeline_loop_no_softmax_ms"] = (
        results["pt_qkv_ms"] + results["pt_dot_loop_ms"] + results["pt_av_loop_ms"]
    )
    results["tvm_pipeline_loop_no_softmax_ms"] = (
        results["tvm_qkv_ms"] + results["tvm_dot_loop_ms"] + results["tvm_av_loop_ms"]
    )

    # Optional estimate with identical softmax term added to both sides.
    results["pt_pipeline_loop_with_softmax_ms"] = (
        results["pt_pipeline_loop_no_softmax_ms"] + results["pt_softmax_ms"]
    )
    results["tvm_pipeline_loop_with_softmax_ms"] = (
        results["tvm_pipeline_loop_no_softmax_ms"] + results["pt_softmax_ms"]
    )

    results["qkv_speedup_pt_over_tvm"] = results["pt_qkv_ms"] / results["tvm_qkv_ms"]
    results["dot_loop_speedup_pt_over_tvm"] = results["pt_dot_loop_ms"] / results["tvm_dot_loop_ms"]
    results["av_loop_speedup_pt_over_tvm"] = results["pt_av_loop_ms"] / results["tvm_av_loop_ms"]
    results["pipeline_no_softmax_speedup_pt_over_tvm"] = (
        results["pt_pipeline_loop_no_softmax_ms"] / results["tvm_pipeline_loop_no_softmax_ms"]
    )

    print(
        f"  QKV: PT {results['pt_qkv_ms']:.4f} ms vs TVM {results['tvm_qkv_ms']:.4f} ms "
        f"({results['qkv_speedup_pt_over_tvm']:.2f}x PT/TVM)"
    )
    print(
        f"  QK loop: PT {results['pt_dot_loop_ms']:.4f} ms vs TVM {results['tvm_dot_loop_ms']:.4f} ms "
        f"({results['dot_loop_speedup_pt_over_tvm']:.2f}x PT/TVM)"
    )
    print(
        f"  AV loop: PT {results['pt_av_loop_ms']:.4f} ms vs TVM {results['tvm_av_loop_ms']:.4f} ms "
        f"({results['av_loop_speedup_pt_over_tvm']:.2f}x PT/TVM)"
    )

    return results


@app.local_entrypoint()
def main(max_trials: int = 800, iters: int = 200, pytorch_only: bool = False):
    import json

    print("=" * 80)
    print("TVM vs PyTorch Attention Benchmark v2 (Fairness-Focused)")
    print("=" * 80)
    print(f"Shapes: {ATTENTION_SHAPES}")
    print(f"Trials: {max_trials} per kernel")
    print(f"Iters : {iters}")
    print(f"PyTorch only: {pytorch_only}")
    print("=" * 80)

    all_results = list(
        tune_and_benchmark_attention_v2.starmap(
            [(s, h, n, m, max_trials, iters, pytorch_only) for s, h, n, m in ATTENTION_SHAPES]
        )
    )

    has_tvm = all_results and "tvm_qkv_ms" in all_results[0]
    if has_tvm:
        print("\n" + "=" * 120)
        print("MATCHED (PER-HEAD LOOP) COMPARISON")
        print("=" * 120)
        hdr = (
            f"{'Model':<7} {'Seq':>5} "
            f"{'PT QKV':>8} {'TVM QKV':>8} {'PT/TVM':>8} "
            f"{'PT QK(loop)':>12} {'TVM QK(loop)':>13} {'PT/TVM':>8} "
            f"{'PT AV(loop)':>12} {'TVM AV(loop)':>13} {'PT/TVM':>8} "
            f"{'PT Pipe(no sfmx)':>16} {'TVM Pipe(no sfmx)':>17} {'PT/TVM':>8}"
        )
        print(hdr)
        print("-" * len(hdr))

        for r in all_results:
            print(
                f"{r['model']:<7} {r['seq_len']:>5} "
                f"{r['pt_qkv_ms']:>8.3f} {r['tvm_qkv_ms']:>8.3f} {r['qkv_speedup_pt_over_tvm']:>7.2f}x "
                f"{r['pt_dot_loop_ms']:>12.3f} {r['tvm_dot_loop_ms']:>13.3f} {r['dot_loop_speedup_pt_over_tvm']:>7.2f}x "
                f"{r['pt_av_loop_ms']:>12.3f} {r['tvm_av_loop_ms']:>13.3f} {r['av_loop_speedup_pt_over_tvm']:>7.2f}x "
                f"{r['pt_pipeline_loop_no_softmax_ms']:>16.3f} {r['tvm_pipeline_loop_no_softmax_ms']:>17.3f} "
                f"{r['pipeline_no_softmax_speedup_pt_over_tvm']:>7.2f}x"
            )
    else:
        print("\n[info] Skipping MATCHED PT-vs-TVM table (pytorch_only mode)")

    print("\n" + "=" * 105)
    print("PYTORCH BEST-PATH REFERENCE")
    print("=" * 105)
    hdr2 = (
        f"{'Model':<7} {'Seq':>5} {'PT Dot(bmm)':>12} {'PT Dot(loop)':>13} "
        f"{'PT AV(bmm)':>11} {'PT AV(loop)':>12} {'PT Full Unfused':>15} {'PT SDPA':>10}"
    )
    print(hdr2)
    print("-" * len(hdr2))

    for r in all_results:
        print(
            f"{r['model']:<7} {r['seq_len']:>5} "
            f"{r['pt_dot_batched_ms']:>12.3f} {r['pt_dot_loop_ms']:>13.3f} "
            f"{r['pt_av_batched_ms']:>11.3f} {r['pt_av_loop_ms']:>12.3f} "
            f"{r['pt_full_unfused_batched_ms']:>15.3f} {r['pt_sdpa_ms']:>10.3f}"
        )

    out_path = Path(__file__).parent.parent / "results" / "attention_results_v2.json"

    if pytorch_only:
        pt_keys = {
            "gpu",
            "arch",
            "pytorch_only",
            "pt_qkv_ms",
            "pt_dot_batched_ms",
            "pt_dot_loop_ms",
            "pt_softmax_ms",
            "pt_av_batched_ms",
            "pt_av_loop_ms",
            "pt_full_unfused_batched_ms",
            "pt_full_unfused_loop_ms",
            "pt_sdpa_ms",
            "pt_pipeline_loop_no_softmax_ms",
            "pt_pipeline_loop_with_softmax_ms",
        }

        try:
            with open(out_path, "r") as f:
                existing_results = json.load(f)
        except FileNotFoundError:
            existing_results = []

        index = {
            (
                r.get("model"),
                r.get("seq_len"),
                r.get("hidden_dim"),
                r.get("num_heads"),
            ): r
            for r in existing_results
        }

        for r in all_results:
            key = (
                r.get("model"),
                r.get("seq_len"),
                r.get("hidden_dim"),
                r.get("num_heads"),
            )
            if key in index:
                dst = index[key]
                for k in pt_keys:
                    if k in r:
                        dst[k] = r[k]
            else:
                index[key] = r

        merged_results = sorted(
            index.values(),
            key=lambda x: (
                str(x.get("model", "")),
                int(x.get("seq_len", 0)),
                int(x.get("hidden_dim", 0)),
                int(x.get("num_heads", 0)),
            ),
        )
        with open(out_path, "w") as f:
            json.dump(merged_results, f, indent=2)
        print(f"\nSaved: {out_path} (PyTorch fields merged; TVM fields preserved)")
    else:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out_path}")
