import modal

app = modal.App("tvm-gemm-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "build-essential"])
    .pip_install([
        "torch",
        "numpy==1.26.4",
        "apache-tvm",
        "decorator",
        "attrs",
    ])
)

# Cache compiled TVM kernels within the container process
_TVM_KERNEL_CACHE = {}


def _build_tvm_gemm(M: int, N: int, K: int, arch: str):
    import tvm
    from tvm import te

    cache_key = (M, N, K, arch)
    if cache_key in _TVM_KERNEL_CACHE:
        return _TVM_KERNEL_CACHE[cache_key]

    # A: (M, K), B: (K, N), C: (M, N)
    A = te.placeholder((M, K), name="A", dtype="float16")
    B = te.placeholder((K, N), name="B", dtype="float16")
    k = te.reduce_axis((0, K), name="k")

    # Accumulate in fp32, cast output back to fp16
    C_fp32 = te.compute(
    (M, N),
        lambda i, j: te.sum(
            A[i, k].astype("float32") * B[k, j].astype("float32"),
            axis=k,
        ),
        name="C_fp32",
    )

    # Stage 2: cast to fp16
    C = te.compute(
        (M, N),
        lambda i, j: C_fp32[i, j].astype("float16"),
        name="C",
    )

    s = te.create_schedule([C_fp32.op, C.op])

    # Simple CUDA schedule
    block_m = 16
    block_n = 16
    thread_m = 8
    thread_n = 8

    i, j = s[C_fp32].op.axis
    ko, ki = s[C_fp32].split(k, factor=8)
    io, ii = s[C_fp32].split(i, factor=block_m)
    jo, ji = s[C_fp32].split(j, factor=block_n)

    iio, iii = s[C_fp32].split(ii, factor=thread_m)
    jio, jij = s[C_fp32].split(ji, factor=thread_n)

    s[C_fp32].reorder(io, jo, ko, iio, jio, ki, iii, jij)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    s[C_fp32].bind(io, block_y)
    s[C_fp32].bind(jo, block_x)
    s[C_fp32].bind(iio, thread_y)
    s[C_fp32].bind(jio, thread_x)

    s[C].compute_inline()

    target = tvm.target.Target(f"cuda -arch={arch}")
    func = tvm.build(s, [A, B, C], target=target, name=f"gemm_{M}_{N}_{K}")

    _TVM_KERNEL_CACHE[cache_key] = func
    return func


@app.function(gpu="B200:1", image=image, timeout=3600)
def benchmark_gemm(M=512, N=768, K=768, model="BERT"):
    import time
    import torch
    import tvm

    device = torch.cuda.get_device_properties(0)
    arch = f"sm_{device.major}{device.minor}"

    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"Detected arch: {arch}")
    print(f"Model: {model} | Shape: ({M}, {K}) x ({K}, {N})")

    # -------------------------
    # PyTorch baseline
    # -------------------------
    A_torch = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B_torch = torch.randn(K, N, dtype=torch.float16, device="cuda")

    for _ in range(10):
        _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - t0) / iters * 1000

    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()
    pytorch_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"[PyTorch] Latency: {pytorch_ms:.3f} ms")
    print(f"[PyTorch] Memory:  {pytorch_memory_mb:.1f} MB")

    # -------------------------
    # TVM compiled kernel
    # -------------------------
    tvm_func = _build_tvm_gemm(M, N, K, arch)
    dev = tvm.cuda(0)

    A_np = A_torch.detach().cpu().numpy()
    B_np = B_torch.detach().cpu().numpy()

    A_tvm = tvm.nd.array(A_np, device=dev)
    B_tvm = tvm.nd.array(B_np, device=dev)
    C_tvm = tvm.nd.empty((M, N), device=dev, dtype="float16")

    for _ in range(10):
        tvm_func(A_tvm, B_tvm, C_tvm)
    dev.sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        tvm_func(A_tvm, B_tvm, C_tvm)
    dev.sync()
    tvm_ms = (time.perf_counter() - t0) / iters * 1000

    # Explicit buffer footprint only
    tvm_memory_mb = (A_np.nbytes + B_np.nbytes + (M * N * 2)) / 1024 / 1024

    print(f"[TVM]     Latency: {tvm_ms:.3f} ms")
    print(f"[TVM]     Memory:  {tvm_memory_mb:.1f} MB (explicit buffers only)")

    return {
        "model": model,
        "seq_len": M,
        "shape": (M, N, K),
        "pytorch_ms": pytorch_ms,
        "pytorch_memory_mb": pytorch_memory_mb,
        "tvm_ms": tvm_ms,
        "tvm_memory_mb": tvm_memory_mb,
        "arch": arch,
    }


@app.local_entrypoint()
def main():
    shapes = [
        # (M=seq_len, N,    K,    model)
        (128,  768,  768,  "BERT"),
        (512,  768,  768,  "BERT"),
        (128,  4096, 4096, "Llama"),
        (512,  4096, 4096, "Llama"),
        (2048, 4096, 4096, "Llama"),
        (4096, 4096, 4096, "Llama"),
    ]

    results = []
    for M, N, K, model in shapes:
        results.append(benchmark_gemm.remote(M, N, K, model))

    print(
        f"\n{'Model':<8} {'Seq Len':<10} "
        f"{'PyTorch (ms)':<15} {'TVM (ms)':<12} "
        f"{'PT Mem (MB)':<12} {'TVM Mem (MB)':<14} {'Arch'}"
    )
    print("-" * 95)

    for r in results:
        print(
            f"{r['model']:<8} {r['seq_len']:<10} "
            f"{r['pytorch_ms']:<15.3f} {r['tvm_ms']:<12.3f} "
            f"{r['pytorch_memory_mb']:<12.1f} {r['tvm_memory_mb']:<14.1f} "
            f"{r['arch']}"
        )