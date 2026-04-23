"""
benchmark_sota_attention.py
==========================
Comprehensive benchmark comparing:
1. Eager PyTorch (Decomposed)
2. PyTorch SDPA (Dispatcher)
3. FlashAttention-2 (SOTA Fused)
4. TVM Decomposed (Your existing pipeline)

Usage:
    modal run benchmark_sota_attention.py
"""

import modal
import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import os

try:
    from app import image
except ImportError:
    image = None

app = modal.App("tvm-sota-benchmark")
volume = modal.Volume.from_name("tvm-tuning-results", create_if_missing=True)

# ── Benchmark Parameters ──────────────────────────────────────────────────────
WORKLOADS = [
    # (seq_len, hidden_dim, num_heads, causal, model_name)
    (128,  768,  12, False, "BERT-128"),
    (512,  768,  12, False, "BERT-512"),
    (128,  4096, 32, True,  "Llama-128"),
    (512,  4096, 32, True,  "Llama-512"),
    (2048, 4096, 32, True,  "Llama-2048"),
]

# ── Helper: Timing ────────────────────────────────────────────────────────────
def time_fn(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

# ── Modal Function ────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="L40S",
    volumes={"/tuning_results": volume},
    timeout=3600,
)
def run_benchmark():
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        from flash_attn import flash_attn_func
        HAS_FLASH = True
    except ImportError:
        HAS_FLASH = False
        print("[WARN] flash_attn not installed. Falling back to SDPA implementation.")

    results_all = []

    for seq_len, hidden_dim, num_heads, causal, name in WORKLOADS:
        print(f"\n>>> Benchmarking {name} (seq={seq_len}, hidden={hidden_dim}, heads={num_heads}, causal={causal})")
        
        batch_size = 1
        head_dim = hidden_dim // num_heads
        dtype = torch.float16
        device = "cuda"
        
        # Input tensors
        X  = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)
        Wq = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
        Wk = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
        Wv = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
        
        # 1. Eager PyTorch (Fully Decomposed)
        def run_eager():
            Q = (X @ Wq).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            K = (X @ Wk).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            V = (X @ Wv).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            if causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V)
            return out.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        eager_ms = time_fn(run_eager)
        
        # 2. PyTorch SDPA (Dispatcher)
        # Force FlashAttention if available via SDPA
        def run_sdpa():
            Q = (X @ Wq).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            K = (X @ Wk).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            V = (X @ Wv).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)

        sdpa_ms = time_fn(run_sdpa)

        # 3. FlashAttention-2 (SOTA Fused via SDPA)
        # We explicitly force the flash backend to ensure we are testing the SOTA path.
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                flash_ms = time_fn(run_sdpa)
        except Exception as e:
            print(f"[WARN] FlashAttention-2 not available for {name}: {e}")
            flash_ms = float('nan')

        # 4. Memory Efficient Attention (xformers/cutlass via SDPA)
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                mem_eff_ms = time_fn(run_sdpa)
        except Exception as e:
            print(f"[WARN] MemoryEfficientAttention not available for {name}: {e}")
            mem_eff_ms = float('nan')

        # 4. TVM Decomposed (Loading existing or estimating based on previous results)
        # For a fair comparison, we use the values from the previous benchmark session
        # or load from the volume if they exist.
        # Here we'll look for 'attention_results.json' in the volume.
        tvm_ms = float('nan')
        try:
            with open("/tuning_results/attention_results.json", "r") as f:
                attn_data = json.load(f)
                match = next((item for item in attn_data if item["model"].startswith(name.split('-')[0]) and item["seq_len"] == seq_len), None)
                if match:
                    tvm_ms = match["tvm_full_ms"]
        except Exception as e:
            print(f"[INFO] Could not load TVM results for {name}: {e}")

        # Peak Memory Measurement
        torch.cuda.reset_peak_memory_stats()
        run_eager()
        mem_eager = torch.cuda.max_memory_allocated() / (1024**2)
        
        torch.cuda.reset_peak_memory_stats()
        run_sdpa()
        mem_fused = torch.cuda.max_memory_allocated() / (1024**2)

        results_all.append({
            "workload": name,
            "seq_len": seq_len,
            "eager_ms": eager_ms,
            "sdpa_ms": sdpa_ms,
            "flash_ms": flash_ms,
            "mem_eff_ms": mem_eff_ms,
            "tvm_ms": tvm_ms,
            "mem_eager_mb": mem_eager,
            "mem_fused_mb": mem_fused,
            "speedup_vs_eager": eager_ms / flash_ms if not np.isnan(flash_ms) else 0,
            "speedup_vs_tvm": tvm_ms / flash_ms if not (np.isnan(tvm_ms) or np.isnan(flash_ms)) else 0,
        })

    # Summary and Visualization
    df = pd.DataFrame(results_all)
    print("\n" + "="*80)
    print("FINAL BENCHMARK RESULTS (LATENCY IN MS)")
    print("="*80)
    print(df.to_string(index=False))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(WORKLOADS))
    width = 0.2
    
    ax.bar(x - 1.5*width, df['eager_ms'], width, label='Eager PyTorch (O(N^2))')
    ax.bar(x - 0.5*width, df['tvm_ms'], width, label='TVM Decomposed')
    ax.bar(x + 0.5*width, df['mem_eff_ms'], width, label='MemEff Attention')
    ax.bar(x + 1.5*width, df['flash_ms'], width, label='FlashAttention-2 (SOTA)')
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Attention Latency comparison (L40S, FP16)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['workload'])
    ax.legend()
    ax.set_yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plot_path = "/tuning_results/sota_comparison.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    
    return results_all

@app.local_entrypoint()
def main():
    results = run_benchmark.remote()
    with open("sota_benchmark_final.json", "w") as f:
        json.dump(results, f, indent=2)
