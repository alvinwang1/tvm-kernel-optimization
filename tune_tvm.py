import modal

app = modal.App("tvm-tuning")

# Persistent storage — tuning results survive between runs
volume = modal.Volume.from_name("tvm-tuning-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch==2.1.0",
        "numpy==1.26.4",   # must pin to 1.x — TVM breaks with NumPy 2.0
        "decorator",
        "attrs",
        "cloudpickle",
        "psutil",
        "scipy",
        "xgboost",         # TVM uses this for its cost model
    ])
    .run_commands([
        "pip install apache-tvm",
    ])
)

# ── Shapes to tune ──────────────────────────────────────────────────────────
# (M, N, K) — C = A @ B where A is (M,K) and B is (K,N)
# M = sequence length, N/K = hidden dimension
GEMM_SHAPES = [
    (128,  768,  768),    # BERT, seq_len=128
    (512,  768,  768),    # BERT, seq_len=512
    (128,  4096, 4096),   # Llama, seq_len=128
    (512,  4096, 4096),   # Llama, seq_len=512
    (2048, 4096, 4096),   # Llama, seq_len=2048
]


@app.function(
    gpu="A100",            # A100 has best TVM support — don't use B200
    image=image,
    timeout=7200,          # 2 hours — tuning is slow
    volumes={"/tuning_results": volume},
)
def tune_gemm(M: int, N: int, K: int, max_trials: int = 800):
    """
    Autotune a single GEMM kernel using TVM MetaSchedule.
    Saves the tuning database to the Modal Volume.
    """
    import tvm
    import tvm.te as te
    from tvm import meta_schedule as ms
    import torch

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n{'='*60}")
    print(f"Tuning GEMM ({M}, {K}) x ({K}, {N}) on {gpu_name}")
    print(f"Max trials: {max_trials}")
    print(f"{'='*60}\n")

    # This tells TVM max_threads_per_block and other GPU-specific info
    # that MetaSchedule needs to generate valid kernels
    target = tvm.target.Target("nvidia/nvidia-a100")

    # ── Define the GEMM computation ──────────────────────────────────────────
    k_axis = te.reduce_axis((0, K), name="k")
    A = te.placeholder((M, K), name="A", dtype="float16")
    B = te.placeholder((K, N), name="B", dtype="float16")
    # matrix multiply: C[i,j] = sum over k of A[i,k] * B[k,j]
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k_axis] * B[k_axis, j], axis=k_axis),
        name="C",
    )

    func = te.create_prim_func([A, B, C])
    mod = tvm.IRModule({"main": func})

    work_dir = f"/tuning_results/gemm_{M}_{N}_{K}"
    print(f"Saving results to: {work_dir}")

    # ── Run autotuning ───────────────────────────────────────────────────────
    # TVM tries `max_trials` different kernel implementations,
    # measures each on real hardware, and keeps the best one.
    database = ms.tune_tir(
        mod=mod,
        target=target,
        max_trials_global=max_trials,
        num_trials_per_iter=64,
        work_dir=work_dir,
    )

    # Compile the best found schedule to check it worked
    sch = ms.tir_integration.compile_tir(database, mod, target)
    if sch is None:
        print("WARNING: No valid schedule found!")
        return None

    print(f"\nTuning complete for shape ({M},{N},{K})!")
    volume.commit()  # save to Modal Volume
    return work_dir


@app.local_entrypoint()
def main():
    print("Starting TVM GEMM tuning jobs...")
    print(f"Shapes to tune: {GEMM_SHAPES}\n")

    # Submit all shapes as parallel jobs
    results = list(tune_gemm.starmap(GEMM_SHAPES))

    print("\n" + "="*60)
    print("ALL TUNING COMPLETE")
    print("="*60)
    for shape, result in zip(GEMM_SHAPES, results):
        status = "PASSED" if result else "FAILED"
        print(f"  {status}  Shape {shape} → {result}")
    print("\nRun `modal run benchmark.py` to see results!")