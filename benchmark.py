"""
benchmark.py — Run this to compare TVM vs PyTorch vs cuBLAS.
Usage: modal run benchmark.py

Run tune.py first to generate TVM kernels.
This script loads those saved kernels and benchmarks everything side by side.
"""

import modal

app = modal.App("tvm-benchmark")

volume = modal.Volume.from_name("tvm-tuning-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch==2.1.0",
        "numpy",
        "decorator",
        "attrs",
        "cloudpickle",
        "psutil",
        "scipy",
        "xgboost",
        "tabulate",  # pretty-print results table
    ])
    .run_commands([
        "pip install apache-tvm",
    ])
)

GEMM_SHAPES = [
    (128,  768,  768),
    (512,  768,  768),
    (128,  4096, 4096),
    (512,  4096, 4096),
    (2048, 4096, 4096),
]


def benchmark_torch(A, B, iters=200):
    """Benchmark PyTorch matmul (uses cuBLAS under the hood)."""
    import torch
    import time

    # warmup — GPU needs a few runs to reach peak performance
    for _ in range(20):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / iters * 1000  # milliseconds


def benchmark_tvm_kernel(lib, A_tvm, B_tvm, C_tvm, iters=200):
    """Benchmark a compiled TVM kernel."""
    import time
    import tvm

    dev = tvm.cuda(0)

    # warmup
    for _ in range(20):
        lib(A_tvm, B_tvm, C_tvm)
    dev.sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        lib(A_tvm, B_tvm, C_tvm)
    dev.sync()

    return (time.perf_counter() - t0) / iters * 1000  # milliseconds


def compute_tflops(M, N, K, time_ms):
    """
    How many trillion floating point ops per second?
    GEMM does 2*M*N*K operations (multiply + add for each element).
    Higher = better.
    """
    flops = 2 * M * N * K
    tflops = flops / (time_ms / 1000) / 1e12
    return tflops


@app.function(
    gpu="A100",
    image=image,
    timeout=3600,
    volumes={"/tuning_results": volume},
)
def run_benchmark(M: int, N: int, K: int):
    """
    For one GEMM shape:
      1. Benchmark PyTorch (cuBLAS)
      2. Load saved TVM kernel and benchmark it
      3. Return comparison results
    """
    import torch
    import numpy as np
    import tvm
    from tvm import meta_schedule as ms
    import tvm.te as te

    gpu_name = torch.cuda.get_device_name(0)
    target = tvm.target.cuda(arch="sm_80")
    dev = tvm.cuda(0)

    print(f"\nBenchmarking shape ({M}, {K}) x ({K}, {N}) on {gpu_name}")

    results = {
        "shape": f"({M},{N},{K})",
        "M": M, "N": N, "K": K,
    }

    # ── 1. PyTorch / cuBLAS benchmark ───────────────────────────────────────
    A_torch = torch.randn(M, K, dtype=torch.float16).cuda()
    B_torch = torch.randn(K, N, dtype=torch.float16).cuda()

    pytorch_ms = benchmark_torch(A_torch, B_torch)
    results["pytorch_ms"] = pytorch_ms
    results["pytorch_tflops"] = compute_tflops(M, N, K, pytorch_ms)
    print(f"  PyTorch (cuBLAS): {pytorch_ms:.3f} ms  |  {results['pytorch_tflops']:.2f} TFLOPS")

    # ── 2. TVM tuned kernel benchmark ───────────────────────────────────────
    work_dir = f"/tuning_results/gemm_{M}_{N}_{K}"

    try:
        # Rebuild the same computation graph
        k_axis = te.reduce_axis((0, K), name="k")
        A_ph = te.placeholder((M, K), name="A", dtype="float16")
        B_ph = te.placeholder((K, N), name="B", dtype="float16")
        C_ph = te.compute(
            (M, N),
            lambda i, j: te.sum(A_ph[i, k_axis] * B_ph[k_axis, j], axis=k_axis),
            name="C",
        )

        func = te.create_prim_func([A_ph, B_ph, C_ph])
        mod = tvm.IRModule({"main": func})

        # Load the tuning database
        database = ms.database.JSONDatabase(work_dir=work_dir)
        sch = ms.tir_integration.compile_tir(database, mod, target)

        if sch is None:
            raise RuntimeError("No valid schedule in database")

        # Compile to a runnable kernel
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(sch.mod, target=target)

        # Create TVM tensors from numpy (TVM needs its own tensor format)
        A_np = A_torch.cpu().numpy()
        B_np = B_torch.cpu().numpy()
        C_np = np.zeros((M, N), dtype="float16")

        A_tvm = tvm.nd.array(A_np, dev)
        B_tvm = tvm.nd.array(B_np, dev)
        C_tvm = tvm.nd.array(C_np, dev)

        tvm_ms = benchmark_tvm_kernel(lib, A_tvm, B_tvm, C_tvm)
        results["tvm_ms"] = tvm_ms
        results["tvm_tflops"] = compute_tflops(M, N, K, tvm_ms)
        results["speedup_vs_pytorch"] = pytorch_ms / tvm_ms
        print(f"  TVM (autotuned):  {tvm_ms:.3f} ms  |  {results['tvm_tflops']:.2f} TFLOPS  |  {results['speedup_vs_pytorch']:.2f}x vs PyTorch")

        # ── 3. Correctness check ─────────────────────────────────────────────
        # Make sure TVM gives the same answer as PyTorch
        C_torch = torch.matmul(A_torch, B_torch).cpu().numpy()
        C_tvm_np = C_tvm.numpy()
        max_diff = np.max(np.abs(C_torch - C_tvm_np))
        results["max_diff"] = float(max_diff)
        results["correct"] = bool(max_diff < 0.1)  # FP16 has some tolerance
        print(f"  Correctness check: max_diff={max_diff:.4f} → {'PASS ✓' if results['correct'] else 'FAIL ✗'}")

    except Exception as e:
        print(f"  TVM kernel not available: {e}")
        print(f"  (Run tune.py first to generate tuned kernels)")
        results["tvm_ms"] = None
        results["tvm_tflops"] = None
        results["speedup_vs_pytorch"] = None

    return results


@app.local_entrypoint()
def main():
    from tabulate import tabulate

    print("="*70)
    print("TVM vs PyTorch GEMM Benchmark")
    print("="*70)

    # Run all shapes in parallel
    all_results = list(run_benchmark.starmap(GEMM_SHAPES))

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    table_rows = []
    for r in all_results:
        tvm_ms   = f"{r['tvm_ms']:.3f}"   if r["tvm_ms"]   else "N/A (tune first)"
        tvm_tf   = f"{r['tvm_tflops']:.2f}" if r["tvm_tflops"] else "—"
        speedup  = f"{r['speedup_vs_pytorch']:.2f}x" if r["speedup_vs_pytorch"] else "—"
        correct  = "✓" if r.get("correct") else "—"

        table_rows.append([
            r["shape"],
            f"{r['pytorch_ms']:.3f}",
            f"{r['pytorch_tflops']:.2f}",
            tvm_ms,
            tvm_tf,
            speedup,
            correct,
        ])

    headers = [
        "Shape (M,N,K)",
        "PyTorch (ms)",
        "PyTorch (TFLOPS)",
        "TVM (ms)",
        "TVM (TFLOPS)",
        "TVM Speedup",
        "Correct?",
    ]

    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))
    print("\nNote: Higher TFLOPS = better. Speedup > 1x means TVM is faster.")

    # ── Save raw results ─────────────────────────────────────────────────────
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nRaw results saved to benchmark_results.json")