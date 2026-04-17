import modal

app = modal.App("tvm-gemm-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["torch", "numpy==1.26.4", "apache-tvm", "decorator", "attrs"])
)

@app.function(gpu="B200:1", image=image, timeout=3600)
def benchmark_gemm(M=512, N=768, K=768, model="BERT"):
    import torch
    import time

    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model} | Shape: ({M}, {K}) x ({K}, {N})")

    A = torch.randn(M, K, dtype=torch.float16).cuda()
    B = torch.randn(K, N, dtype=torch.float16).cuda()

    # warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # latency benchmark
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - t0) / iters * 1000

    # memory benchmark
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Latency:  {pytorch_ms:.3f} ms")
    print(f"  Memory:   {memory_mb:.1f} MB")

    return {
        "model": model,
        "seq_len": M,        # seq len
        "shape": (M, N, K),
        "pytorch_ms": pytorch_ms,
        "memory_mb": memory_mb,
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
        result = benchmark_gemm.remote(M, N, K, model)
        results.append(result)

    # ── Print results table ───────────────────────────────
    print(f"\n{'Model':<8} {'Seq Len':<10} {'Latency (ms)':<15} {'Memory (MB)'}")
    print("-" * 50)
    for r in results:
        print(f"{r['model']:<8} {r['seq_len']:<10} {r['pytorch_ms']:<15.3f} {r['memory_mb']:.1f}")