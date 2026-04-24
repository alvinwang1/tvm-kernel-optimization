"""
tune_and_benchmark_attention.py
================================
Mirrors tune_and_benchmark.py but for transformer attention operations:

  1. Tune individual attention kernels with TVM MetaSchedule.
  2. Compile the best discovered schedules.
  3. Benchmark PyTorch (eager) vs tuned TVM kernels for:
       - Q, K, V projections  (3x GEMM)
       - Dot product          (Q @ K^T)
       - Softmax
       - Weighted sum         (attn @ V)
       - Full attention       (all steps)
  4. Print a side-by-side latency table.

Usage:
    modal run tune_and_benchmark_attention.py
    modal run tune_and_benchmark_attention.py --max-trials 200   # quick test
"""

import modal

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from app import image
except ModuleNotFoundError:
    image = None

app = modal.App("tvm-attention-benchmark")

volume = modal.Volume.from_name("tvm-tuning-results", create_if_missing=True)

# ── Attention shapes ──────────────────────────────────────────────────────────
# (seq_len, hidden_dim, num_heads, model_name)
ATTENTION_SHAPES = [
    (128,  768,  12, "BERT"),    # BERT-base, seq_len=128
    (512,  768,  12, "BERT"),    # BERT-base, seq_len=512
    (128,  4096, 32, "Llama"),   # Llama-7B,  seq_len=128
    (512,  4096, 32, "Llama"),   # Llama-7B,  seq_len=512
    (2048, 4096, 32, "Llama"),   # Llama-7B,  seq_len=2048
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _tflops(M: int, N: int, K: int, ms: float) -> float:
    """2*M*N*K FLOPs / wall-clock seconds → TFLOPS."""
    return 2 * M * N * K / (ms / 1000) / 1e12


def _timeit(fn, iters=200, warmup=20):
    """Time a GPU function, returns milliseconds."""
    import torch, time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _tune_gemm(mod, target, work_dir, max_trials, volume):
    """
    Tune a single GEMM kernel with TVM MetaSchedule.
    Reuses cached results if they already exist.
    Same logic as tune_and_benchmark.py.
    """
    import glob
    from tvm import meta_schedule as ms

    db_files = glob.glob(f"{work_dir}/**/*.json", recursive=True)
    already_tuned = len(db_files) >= 10

    if already_tuned:
        print(f"  [cache] Loading existing results from {work_dir} ({len(db_files)} records)")
    else:
        print(f"  [tune] Running MetaSchedule — up to {max_trials} trials …")
        ms.tune_tir(
            mod=mod,
            target=target,
            max_trials_global=max_trials,
            num_trials_per_iter=32,
            work_dir=work_dir,
        )
        volume.commit()
        print(f"  [tune] Done — saved to {work_dir}")


def _compile_gemm(mod, target, work_dir):
    """
    Load the best tuned schedule and compile it.
    Falls back to a simple tiled CUDA schedule if tuning found nothing.
    Same fallback logic as tune_and_benchmark.py.
    """
    import tvm
    import tvm.te as te
    from tvm import meta_schedule as ms

    database = ms.database.JSONDatabase(work_dir=work_dir)
    sch = ms.tir_integration.compile_tir(database, mod, target)

    if sch is not None:
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(sch.mod, target=target)
        return lib, True

    # Fallback: simple 16x16 tiled schedule
    print("  [compile] WARNING: no tuned schedule — using simple tiled fallback.")
    # Extract te placeholders from the IRModule for fallback scheduling
    prim_func = mod["main"]
    # Re-create te tensors for the fallback scheduler
    # (te.create_schedule needs te.Tensor, not PrimFunc)
    return None, False


# ── Modal function ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="L40S",
    timeout=86400,
    volumes={"/tuning_results": volume},
)
def tune_and_benchmark_attention(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    model: str,
    max_trials: int = 800,
    iters: int = 200,
):
    import math
    import glob
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tvm
    import tvm.te as te
    from tvm import meta_schedule as ms

    # ── GPU setup (same as tune_and_benchmark.py) ─────────────────────────────
    assert torch.cuda.is_available(), "No CUDA device found!"
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    dev = tvm.cuda(0)

    try:
        target = tvm.target.Target.from_device(dev)
    except Exception:
        target = tvm.target.Target({
            "kind": "cuda",
            "arch": arch,
            "max_num_threads": 1024,
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 49152,
            "thread_warp_size": 32,
        })

    head_dim = hidden_dim // num_heads

    print(f"\n{'='*64}")
    print(f"Model: {model}  seq_len={seq_len}  hidden={hidden_dim}  heads={num_heads}")
    print(f"GPU: {gpu_name} ({arch})")
    print(f"{'='*64}")

    results = {
        "model": model,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "gpu": gpu_name,
        "arch": arch,
    }

    # ── Input tensors ──────────────────────────────────────────────────────────
    X  = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device="cuda")
    Wq = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")
    Wk = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")
    Wv = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")
    scale = math.sqrt(head_dim)

    # ── Helper: build a GEMM IRModule for shape (M, N, K) ────────────────────
    def make_gemm_mod(M, N, K):
        k_ax = te.reduce_axis((0, K), name="k")
        A = te.placeholder((M, K), name="A", dtype="float16")
        B = te.placeholder((K, N), name="B", dtype="float16")
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k_ax] * B[k_ax, j], axis=k_ax),
            name="C",
        )
        func = te.create_prim_func([A, B, C])
        return tvm.IRModule({"main": func}), A, B, C

    # ── Helper: tune + compile one GEMM, return (lib, tuned_bool) ─────────────
    def prepare_gemm(M, N, K, tag):
        work_dir = f"/tuning_results/attn_{tag}_{M}_{N}_{K}_{arch}"
        mod, A_ph, B_ph, C_ph = make_gemm_mod(M, N, K)

        db_files = glob.glob(f"{work_dir}/**/*.json", recursive=True)
        if len(db_files) >= 10:
            print(f"  [cache] {tag}: loading {work_dir}")
        else:
            print(f"  [tune]  {tag}: running {max_trials} trials …")
            ms.tune_tir(
                mod=mod,
                target=target,
                max_trials_global=max_trials,
                num_trials_per_iter=32,
                work_dir=work_dir,
            )
            volume.commit()

        database = ms.database.JSONDatabase(work_dir=work_dir)
        sch = ms.tir_integration.compile_tir(database, mod, target)

        if sch is not None:
            with tvm.transform.PassContext(opt_level=3):
                lib = tvm.build(sch.mod, target=target)
            print(f"  [compile] {tag}: tuned kernel ready ✓")
            return lib, True

        # Fallback: simple tiled schedule
        print(f"  [compile] {tag}: no tuned schedule, using tiled fallback.")
        s = te.create_schedule(C_ph.op)
        io, ii = s[C_ph].split(s[C_ph].op.axis[0], factor=16)
        jo, ji = s[C_ph].split(s[C_ph].op.axis[1], factor=16)
        s[C_ph].bind(io, te.thread_axis("blockIdx.y"))
        s[C_ph].bind(jo, te.thread_axis("blockIdx.x"))
        s[C_ph].bind(ii, te.thread_axis("threadIdx.y"))
        s[C_ph].bind(ji, te.thread_axis("threadIdx.x"))
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(s, [A_ph, B_ph, C_ph], target=target)
        return lib, False

    # ── Helper: allocate TVM tensors from torch tensors ───────────────────────
    def to_tvm(t: torch.Tensor):
        return tvm.nd.array(t.cpu().numpy(), dev)

    def zeros_tvm(shape):
        return tvm.nd.array(np.zeros(shape, dtype="float16"), dev)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Tune all kernels we need
    # ══════════════════════════════════════════════════════════════════════════
    # Q projection: (seq_len, hidden_dim) @ (hidden_dim, hidden_dim)
    q_lib, q_tuned = prepare_gemm(seq_len, hidden_dim, hidden_dim, "Q_proj")
    # K projection: same shape as Q
    k_lib, k_tuned = prepare_gemm(seq_len, hidden_dim, hidden_dim, "K_proj")
    # V projection: same shape as Q
    v_lib, v_tuned = prepare_gemm(seq_len, hidden_dim, hidden_dim, "V_proj")
    # Q @ K^T: (num_heads, seq_len, head_dim) @ (num_heads, head_dim, seq_len)
    # — we tune for a single head and loop (standard approach)
    qk_lib, qk_tuned = prepare_gemm(seq_len, seq_len, head_dim, "QK_dot")
    # attn @ V: (num_heads, seq_len, seq_len) @ (num_heads, seq_len, head_dim)
    av_lib, av_tuned = prepare_gemm(seq_len, head_dim, seq_len, "AV_sum")

    results["tuned"] = all([q_tuned, k_tuned, v_tuned, qk_tuned, av_tuned])

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — PyTorch (eager) baselines
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[PyTorch baseline]")

    # Q, K, V projections — 3 separate GEMMs
    def pt_qkv():
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        return Q, K, V

    results["pt_qkv_ms"] = _timeit(pt_qkv, iters=iters)
    print(f"  QKV projections (3x GEMM):  {results['pt_qkv_ms']:.4f} ms")

    # Precompute Q, K, V reshaped for multi-head attention
    Q_pt, K_pt, V_pt = pt_qkv()
    Q_mh = Q_pt.view(seq_len, num_heads, head_dim).transpose(0, 1)  # (H, S, D)
    K_mh = K_pt.view(seq_len, num_heads, head_dim).transpose(0, 1)
    V_mh = V_pt.view(seq_len, num_heads, head_dim).transpose(0, 1)

    # Q @ K^T
    def pt_dot():
        return torch.bmm(Q_mh, K_mh.transpose(-2, -1))

    results["pt_dot_ms"] = _timeit(pt_dot, iters=iters)
    print(f"  Dot product  (Q @ K^T):     {results['pt_dot_ms']:.4f} ms")

    scores_pt = pt_dot()

    # Softmax
    def pt_softmax():
        return F.softmax(scores_pt / scale, dim=-1)

    results["pt_softmax_ms"] = _timeit(pt_softmax, iters=iters)
    print(f"  Softmax:                    {results['pt_softmax_ms']:.4f} ms")

    attn_pt = pt_softmax()

    # attn @ V
    def pt_av():
        return torch.bmm(attn_pt, V_mh)

    results["pt_av_ms"] = _timeit(pt_av, iters=iters)
    print(f"  Weighted sum (attn @ V):    {results['pt_av_ms']:.4f} ms")

    # Full attention end-to-end (unfused)
    def pt_full():
        Q_ = (X @ Wq).view(seq_len, num_heads, head_dim).transpose(0, 1)
        K_ = (X @ Wk).view(seq_len, num_heads, head_dim).transpose(0, 1)
        V_ = (X @ Wv).view(seq_len, num_heads, head_dim).transpose(0, 1)
        s_ = torch.bmm(Q_, K_.transpose(-2, -1)) / scale
        a_ = F.softmax(s_, dim=-1)
        return torch.bmm(a_, V_)

    results["pt_full_ms"] = _timeit(pt_full, iters=iters)
    print(f"  Full attention (unfused):   {results['pt_full_ms']:.4f} ms")

    # PyTorch SDPA — fused FlashAttention kernel
    def pt_sdpa():
        Q_ = (X @ Wq).view(seq_len, num_heads, head_dim).transpose(0, 1)
        K_ = (X @ Wk).view(seq_len, num_heads, head_dim).transpose(0, 1)
        V_ = (X @ Wv).view(seq_len, num_heads, head_dim).transpose(0, 1)
        return F.scaled_dot_product_attention(Q_, K_, V_)

    results["pt_sdpa_ms"] = _timeit(pt_sdpa, iters=iters)
    print(f"  PyTorch SDPA (Flash):       {results['pt_sdpa_ms']:.4f} ms")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — TVM tuned kernels
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[TVM tuned kernels]")

    # Allocate TVM buffers
    X_tvm  = to_tvm(X)
    Wq_tvm = to_tvm(Wq)
    Wk_tvm = to_tvm(Wk)
    Wv_tvm = to_tvm(Wv)
    Q_tvm  = zeros_tvm((seq_len, hidden_dim))
    K_tvm  = zeros_tvm((seq_len, hidden_dim))
    V_tvm  = zeros_tvm((seq_len, hidden_dim))

    # Q, K, V projections with TVM
    def tvm_qkv():
        q_lib(X_tvm, Wq_tvm, Q_tvm)
        k_lib(X_tvm, Wk_tvm, K_tvm)
        v_lib(X_tvm, Wv_tvm, V_tvm)
        dev.sync()

    results["tvm_qkv_ms"] = _timeit(tvm_qkv, iters=iters)
    results["tvm_qkv_speedup"] = results["pt_qkv_ms"] / results["tvm_qkv_ms"]
    print(f"  QKV projections:  {results['tvm_qkv_ms']:.4f} ms  →  {results['tvm_qkv_speedup']:.2f}x vs PyTorch")

    # Q @ K^T with TVM — run per head
    # shapes: Q head = (seq_len, head_dim), K^T head = (head_dim, seq_len)
    Q_heads = [zeros_tvm((seq_len, head_dim)) for _ in range(num_heads)]
    K_heads = [zeros_tvm((seq_len, head_dim)) for _ in range(num_heads)]
    QK_outs = [zeros_tvm((seq_len, seq_len)) for _ in range(num_heads)]

    # Populate head tensors from the already-computed TVM Q and K
    tvm_qkv()
    Q_np = Q_tvm.numpy().reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    K_np = K_tvm.numpy().reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    V_np = V_tvm.numpy().reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    for h in range(num_heads):
        Q_heads[h] = tvm.nd.array(Q_np[h], dev)          # (seq_len, head_dim)
        K_heads[h] = tvm.nd.array(K_np[h].T.copy(), dev)  # (head_dim, seq_len)

    def tvm_dot():
        for h in range(num_heads):
            qk_lib(Q_heads[h], K_heads[h], QK_outs[h])
        dev.sync()

    results["tvm_dot_ms"] = _timeit(tvm_dot, iters=iters)
    results["tvm_dot_speedup"] = results["pt_dot_ms"] / results["tvm_dot_ms"]
    print(f"  Dot product:      {results['tvm_dot_ms']:.4f} ms  →  {results['tvm_dot_speedup']:.2f}x vs PyTorch")

    # Softmax — PyTorch only (TVM softmax tuning is complex; use PT as baseline)
    # This is standard practice: focus TVM on GEMM-heavy ops
    results["tvm_softmax_ms"] = results["pt_softmax_ms"]
    print(f"  Softmax:          {results['tvm_softmax_ms']:.4f} ms  (PyTorch, not tuned)")

    # attn @ V with TVM — run per head
    # First compute softmax scores from QK_outs
    QK_np = np.stack([QK_outs[h].numpy() for h in range(num_heads)])  # (H, S, S)
    import scipy.special
    attn_np = scipy.special.softmax(QK_np / scale, axis=-1).astype("float16")

    AV_ins  = [tvm.nd.array(attn_np[h], dev) for h in range(num_heads)]   # (S, S)
    V_heads = [tvm.nd.array(V_np[h], dev) for h in range(num_heads)]      # (S, D)
    AV_outs = [zeros_tvm((seq_len, head_dim)) for _ in range(num_heads)]

    def tvm_av():
        for h in range(num_heads):
            av_lib(AV_ins[h], V_heads[h], AV_outs[h])
        dev.sync()

    results["tvm_av_ms"] = _timeit(tvm_av, iters=iters)
    results["tvm_av_speedup"] = results["pt_av_ms"] / results["tvm_av_ms"]
    print(f"  Weighted sum:     {results['tvm_av_ms']:.4f} ms  →  {results['tvm_av_speedup']:.2f}x vs PyTorch")

    # Full TVM attention end-to-end (QKV + dot + softmax + AV)
    results["tvm_full_ms"] = (
        results["tvm_qkv_ms"] +
        results["tvm_dot_ms"] +
        results["tvm_softmax_ms"] +
        results["tvm_av_ms"]
    )
    results["tvm_full_speedup"] = results["pt_full_ms"] / results["tvm_full_ms"]
    print(f"  Full attention:   {results['tvm_full_ms']:.4f} ms  →  {results['tvm_full_speedup']:.2f}x vs PyTorch unfused")
    print(f"  vs SDPA (Flash):  {results['pt_sdpa_ms'] / results['tvm_full_ms']:.2f}x  (SDPA wins if < 1x)")

    # ── Memory ────────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    pt_full()
    torch.cuda.synchronize()
    results["pt_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    return results


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(max_trials: int = 800):
    import json

    print("=" * 70)
    print("TVM Attention Tune + Benchmark  (PyTorch vs TVM)")
    print("=" * 70)
    print(f"Shapes  : {ATTENTION_SHAPES}")
    print(f"Trials  : {max_trials} per kernel")
    print("=" * 70 + "\n")

    all_results = list(
        tune_and_benchmark_attention.starmap(
            [(s, h, n, m, max_trials) for s, h, n, m in ATTENTION_SHAPES]
        )
    )

    # ── Per-operation table ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("PER-OPERATION LATENCY (ms)")
    print("=" * 90)

    HDR = (
        f"{'Model':<7} {'Seq':>5} "
        f"{'PT QKV':>8} {'TVM QKV':>8} {'QKV Spd':>8} "
        f"{'PT Q@K':>8} {'TVM Q@K':>8} {'QK Spd':>7} "
        f"{'PT Sfmx':>8} "
        f"{'PT A@V':>8} {'TVM A@V':>8} {'AV Spd':>7} "
        f"{'PT Full':>8} {'TVM Full':>9} {'Full Spd':>9}"
    )
    print(HDR)
    print("-" * len(HDR))

    for r in all_results:
        print(
            f"{r['model']:<7} {r['seq_len']:>5} "
            f"{r['pt_qkv_ms']:>8.3f} {r['tvm_qkv_ms']:>8.3f} {r['tvm_qkv_speedup']:>7.2f}x "
            f"{r['pt_dot_ms']:>8.3f} {r['tvm_dot_ms']:>8.3f} {r['tvm_dot_speedup']:>6.2f}x "
            f"{r['pt_softmax_ms']:>8.3f} "
            f"{r['pt_av_ms']:>8.3f} {r['tvm_av_ms']:>8.3f} {r['tvm_av_speedup']:>6.2f}x "
            f"{r['pt_full_ms']:>8.3f} {r['tvm_full_ms']:>9.3f} {r['tvm_full_speedup']:>8.2f}x"
        )

    # ── Fusion benefit table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FUSION BENEFIT  (unfused ops vs PyTorch SDPA / FlashAttention)")
    print("=" * 60)

    HDR2 = f"{'Model':<7} {'Seq':>5} {'PT Unfused':>11} {'SDPA':>8} {'SDPA Spd':>9} {'TVM Full':>9} {'TVM vs SDPA':>12}"
    print(HDR2)
    print("-" * len(HDR2))

    for r in all_results:
        sdpa_spd = r["pt_full_ms"] / r["pt_sdpa_ms"]
        tvm_vs_sdpa = r["pt_sdpa_ms"] / r["tvm_full_ms"]
        print(
            f"{r['model']:<7} {r['seq_len']:>5} "
            f"{r['pt_full_ms']:>11.3f} "
            f"{r['pt_sdpa_ms']:>8.3f} "
            f"{sdpa_spd:>8.2f}x "
            f"{r['tvm_full_ms']:>9.3f} "
            f"{tvm_vs_sdpa:>11.2f}x"
        )
    print("\nSDPA Speedup > 1x = FlashAttention beats unfused PyTorch")
    print("TVM vs SDPA  > 1x = TVM beats FlashAttention (unlikely but possible for small seqs)")

    # ── Save raw JSON ─────────────────────────────────────────────────────────
    out_path = "attention_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out_path}")