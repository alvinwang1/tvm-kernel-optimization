"""
tune_and_benchmark.py
=====================
Uses the same production TVM image defined in app.py (CUDA 12.6 + LLVM 18 +
TVM built from source) to:

  1. Tune a float16 GEMM with TVM MetaSchedule on real GPU hardware.
  2. Compile the best discovered schedule.
  3. Benchmark PyTorch (cuBLAS) vs the tuned TVM kernel.
  4. Print a side-by-side latency / TFLOPS table.

Usage:
    modal run tune_and_benchmark.py                        # all shapes
    modal run tune_and_benchmark.py --max-trials 200       # quick smoke-test
"""

import modal
from pathlib import Path

# `from modal_app import image` only runs in the local Python process where app.py
# exists. The remote container already has `image` baked into the function
# decorator — it never re-evaluates this import at runtime.
try:
    from modal_app import image
except ModuleNotFoundError:
    image = None  # remote: image already bound via @app.function(image=...)

# ── App & persistent storage ─────────────────────────────────────────────────
app = modal.App("tvm-tune-and-benchmark")

# Tuning results persist across runs so you don't re-tune every time.
volume = modal.Volume.from_name("tvm-tuning-results", create_if_missing=True)


# ── GEMM shapes to benchmark ──────────────────────────────────────────────────
# (M, N, K)  →  C[M,N] = A[M,K] @ B[K,N]   (fp16)
# M  = sequence length, N/K = hidden dimension
GEMM_SHAPES = [
    (128,   768,  768),   # BERT, seq_len=128
    (512,   768,  768),   # BERT, seq_len=512
    (128,  4096, 4096),   # Llama-7B, seq_len=128
    (512,  4096, 4096),   # Llama-7B, seq_len=512
    (2048, 4096, 4096),   # Llama-7B, seq_len=2048
]


# ── Helper: TFLOPS calculation ────────────────────────────────────────────────
def _tflops(M: int, N: int, K: int, ms: float) -> float:
    """2*M*N*K FLOPs divided by wall-clock time in seconds → TFLOPS."""
    return 2 * M * N * K / (ms / 1000) / 1e12


# ── Modal function ────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="L40S",          # L40S matches app.py; swap for A100/H100 if preferred
    timeout=60 * 90,     # 90 min per shape — tuning is the slow part
    volumes={"/tuning_results": volume},
)
def tune_and_benchmark(
    M: int,
    N: int,
    K: int,
    max_trials: int = 800,
    iters: int = 200,
):
    """
    For a single GEMM shape (M, N, K):
      1. Tune with TVM MetaSchedule (or load cached results).
      2. Compile the best schedule.
      3. Benchmark PyTorch (cuBLAS) vs the tuned TVM kernel.
      4. Verify numerical correctness.
      5. Return a dict with all timings.
    """
    import time
    import numpy as np
    import torch
    import tvm
    import tvm.te as te
    from tvm import meta_schedule as ms

    # ── GPU info ──────────────────────────────────────────────────────────────
    assert torch.cuda.is_available(), "No CUDA device found!"
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    dev = tvm.cuda(0)

    # Target.from_device() queries the live GPU and populates all hardware
    # attributes (max_threads_per_block, max_shared_memory, etc.) that
    # MetaSchedule's AutoBind rule requires. A bare "cuda -arch=sm_XX" string
    # is missing these and causes TuneContext initialization to fail.
    try:
        target = tvm.target.Target.from_device(dev)
    except Exception:
        # Fallback: build the target with explicit thread/memory attributes
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

    print(f"\n{'='*64}")
    print(f"Shape ({M},{K}) x ({K},{N})  —  {gpu_name}  ({arch})")
    print(f"{'='*64}")

    results = {
        "shape": f"({M},{N},{K})",
        "M": M, "N": N, "K": K,
        "gpu": gpu_name, "arch": arch,
    }

    # ── Build TVM IRModule ────────────────────────────────────────────────────
    k_ax = te.reduce_axis((0, K), name="k")
    A_ph = te.placeholder((M, K), name="A", dtype="float16")
    B_ph = te.placeholder((K, N), name="B", dtype="float16")
    C_ph = te.compute(
        (M, N),
        lambda i, j: te.sum(A_ph[i, k_ax] * B_ph[k_ax, j], axis=k_ax),
        name="C",
    )
    func = te.create_prim_func([A_ph, B_ph, C_ph])
    mod = tvm.IRModule({"main": func})

    work_dir = f"/tuning_results/gemm_{M}_{N}_{K}_{arch}"

    # ── Step 1: Tune (or load from cache) ────────────────────────────────────
    import os, glob
    # Check if there's a non-empty database with actual workload records
    db_files = glob.glob(f"{work_dir}/**/*.json", recursive=True)
    # If we have very few records (e.g. < 10), it's likely a failed or interrupted tune.
    already_tuned = len(db_files) >= 10
    if already_tuned:
        print(f"[cache] Loading existing tuning results from {work_dir} ({len(db_files)} records)")
    else:
        print(f"[tune] Running MetaSchedule — up to {max_trials} trials …")
        ms.tune_tir(
            mod=mod,
            target=target,
            max_trials_global=max_trials,
            num_trials_per_iter=32,   # smaller batches → more likely to find valid kernels
            work_dir=work_dir,
        )
        volume.commit()
        print(f"[tune] Done — results saved to {work_dir}")

    # ── Step 2: Compile best schedule ────────────────────────────────────────
    database = ms.database.JSONDatabase(work_dir=work_dir)
    sch = ms.tir_integration.compile_tir(database, mod, target)

    if sch is not None:
        with tvm.transform.PassContext(opt_level=3):
            tvm_lib = tvm.build(sch.mod, target=target)
        print("[compile] TVM tuned kernel compiled successfully.")
        tuned = True
    else:
        # Tuning found no valid schedule — build a minimal CUDA-scheduled baseline
        # using the original te tensors so threads are properly bound.
        print("[compile] WARNING: no tuned schedule found — compiling simple CUDA baseline.")
        s = te.create_schedule(C_ph.op)
        i_ax, j_ax = s[C_ph].op.axis
        # Tile into 16×16 thread blocks
        io, ii = s[C_ph].split(i_ax, factor=16)
        jo, ji = s[C_ph].split(j_ax, factor=16)
        s[C_ph].bind(io, te.thread_axis("blockIdx.y"))
        s[C_ph].bind(jo, te.thread_axis("blockIdx.x"))
        s[C_ph].bind(ii, te.thread_axis("threadIdx.y"))
        s[C_ph].bind(ji, te.thread_axis("threadIdx.x"))
        with tvm.transform.PassContext(opt_level=3):
            tvm_lib = tvm.build(s, [A_ph, B_ph, C_ph], target=target)
        tuned = False

    # ── Step 3: Allocate shared tensors ──────────────────────────────────────
    A_torch = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B_torch = torch.randn(K, N, dtype=torch.float16, device="cuda")

    A_np = A_torch.cpu().numpy()
    B_np = B_torch.cpu().numpy()
    C_np = np.zeros((M, N), dtype="float16")

    A_tvm = tvm.nd.array(A_np, dev)
    B_tvm = tvm.nd.array(B_np, dev)
    C_tvm = tvm.nd.array(C_np, dev)

    # ── Step 4a: Benchmark PyTorch (cuBLAS) ──────────────────────────────────
    # warmup
    for _ in range(20):
        torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - t0) / iters * 1000

    results["pytorch_ms"] = pytorch_ms
    results["pytorch_tflops"] = _tflops(M, N, K, pytorch_ms)
    print(
        f"[PyTorch]  {pytorch_ms:.4f} ms  "
        f"({results['pytorch_tflops']:.2f} TFLOPS)"
    )

    # ── Step 4b: Benchmark TVM tuned kernel ──────────────────────────────────
    # warmup
    for _ in range(20):
        tvm_lib(A_tvm, B_tvm, C_tvm)
    dev.sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        tvm_lib(A_tvm, B_tvm, C_tvm)
    dev.sync()
    tvm_ms = (time.perf_counter() - t0) / iters * 1000

    results["tvm_ms"] = tvm_ms
    results["tvm_tflops"] = _tflops(M, N, K, tvm_ms)
    results["speedup"] = pytorch_ms / tvm_ms
    results["tuned"] = tuned
    label = "tuned" if tuned else "untuned fallback"
    print(
        f"[TVM {label}]  {tvm_ms:.4f} ms  "
        f"({results['tvm_tflops']:.2f} TFLOPS)  "
        f"→  {results['speedup']:.2f}x vs PyTorch"
    )

    # ── Step 5: Numerical correctness ─────────────────────────────────────────
    C_ref = torch.matmul(A_torch, B_torch).cpu().numpy()
    max_diff = float(np.max(np.abs(C_ref - C_tvm.numpy())))
    results["max_diff"] = max_diff
    # fp16 accumulation can have significant drift; 2.0 is usually safe for GEMM
    results["correct"] = max_diff < 2.0   
    status = "PASS ✓" if results["correct"] else "FAIL ✗"
    print(f"[check]    max_diff={max_diff:.4f}  →  {status}")

    return results


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(max_trials: int = 800):
    """
    Tune + benchmark all GEMM shapes in parallel.

    Flags:
      --max-trials INT   MetaSchedule trials per shape (default 800)
                         Use 200 for a quick smoke-test, 1600+ for best perf.
    """
    import json

    print("=" * 70)
    print("TVM MetaSchedule Tune + Benchmark  (PyTorch vs TVM)")
    print("=" * 70)
    print(f"Shapes  : {GEMM_SHAPES}")
    print(f"Trials  : {max_trials} per shape")
    print("=" * 70 + "\n")

    # Run all shapes in parallel on Modal
    all_results = list(
        tune_and_benchmark.starmap(
            [(M, N, K, max_trials) for M, N, K in GEMM_SHAPES]
        )
    )

    # ── Pretty-print results table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    HDR = f"{'Shape':<16} {'PT (ms)':>10} {'PT TFLOPS':>11} {'TVM (ms)':>10} {'TVM TFLOPS':>11} {'Speedup':>9} {'Tuned':>6} {'OK':>4}"
    print(HDR)
    print("-" * len(HDR))
    for r in all_results:
        print(
            f"{r['shape']:<16}"
            f"{r['pytorch_ms']:>10.3f}"
            f"{r['pytorch_tflops']:>11.2f}"
            f"{r['tvm_ms']:>10.3f}"
            f"{r['tvm_tflops']:>11.2f}"
            f"{r['speedup']:>8.2f}x"
            f"  {'✓' if r.get('tuned') else '—':>5}"
            f"  {'✓' if r['correct'] else '✗':>2}"
        )
    print("\nHigher TFLOPS = better.  Speedup > 1x means TVM beats cuBLAS.")

    # ── Save raw JSON ─────────────────────────────────────────────────────────
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out_path}")
