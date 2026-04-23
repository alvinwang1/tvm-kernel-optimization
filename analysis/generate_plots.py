import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from recent benchmarks
data = {
    'Workload': ['BERT-128', 'BERT-512', 'Llama-128', 'Llama-512', 'Llama-2048'],
    'Eager PyTorch (O(N^2))': [0.0746, 0.0781, 0.1452, 0.3377, 4.3402],
    'TVM Decomposed': [0.229, 0.223, 0.645, 0.809, 3.749],
    'Memory Efficient': [0.0452, 0.0439, 0.1189, 0.3095, 1.2861],
    'FlashAttention-2 (SOTA)': [0.0458, 0.0510, 0.1168, 0.2819, 1.1811]
}

df = pd.DataFrame(data)

# Set style
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

x = np.arange(len(df['Workload']))
width = 0.2

# Subplot 1: Latency (Log Scale)
ax1.bar(x - 1.5*width, df['Eager PyTorch (O(N^2))'], width, label='Eager PyTorch', color='#ff7f0e', alpha=0.9)
ax1.bar(x - 0.5*width, df['TVM Decomposed'], width, label='TVM Decomposed', color='#1f77b4', alpha=0.9)
ax1.bar(x + 0.5*width, df['Memory Efficient'], width, label='Mem-Eff Attention', color='#2ca02c', alpha=0.9)
ax1.bar(x + 1.5*width, df['FlashAttention-2 (SOTA)'], width, label='FlashAttention-2', color='#d62728', alpha=1.0)

ax1.set_ylabel('Latency (ms) - Log Scale')
ax1.set_title('Attention Latency comparison (L40S, FP16)')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Workload'])
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Subplot 2: Speedup relative to TVM Decomposed
tvm_baseline = df['TVM Decomposed']
ax2.bar(x - 1.0*width, df['Eager PyTorch (O(N^2))'] / tvm_baseline, width, label='Eager / TVM', color='#ff7f0e', alpha=0.7)
ax2.bar(x, df['Memory Efficient'] / tvm_baseline, width, label='Mem-Eff / TVM', color='#2ca02c', alpha=0.7)
ax2.bar(x + 1.0*width, df['FlashAttention-2 (SOTA)'] / tvm_baseline, width, label='Flash-2 / TVM', color='#d62728', alpha=0.8)

ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='TVM Baseline')
ax2.set_ylabel('Relative Latency (Lower is Better)')
ax2.set_title('SOTA Speedup over TVM Pipeline')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Workload'])
ax2.legend()
ax2.grid(True, axis='y', ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('sota_benchmark_plots.png', dpi=300)
print("Plots generated: sota_benchmark_plots.png")
