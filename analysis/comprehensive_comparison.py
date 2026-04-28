import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Configuration ---
ROOT = Path(__file__).resolve().parent.parent
SOTA_PATH = ROOT / "results" / "sota_benchmark_final.json"
V2_PATH = ROOT / "results" / "attention_results_v2.json"
OUTPUT_PATH = ROOT / "reports" / "comprehensive_comparison.png"

def load_and_merge():
    with open(SOTA_PATH, "r") as f:
        sota_data = json.load(f)
    with open(V2_PATH, "r") as f:
        v2_data = json.load(f)
    
    # Create DataFrames
    df_sota = pd.DataFrame(sota_data)
    df_v2 = pd.DataFrame(v2_data)
    
    # Clean up v2 workload names to match SOTA (BERT-128, etc)
    df_v2['workload'] = df_v2['model'] + "-" + df_v2['seq_len'].astype(str)
    
    # Merge
    merged = []
    for workload in df_sota['workload']:
        s_row = df_sota[df_sota['workload'] == workload].iloc[0]
        v_row = df_v2[df_v2['workload'] == workload].iloc[0]
        
        merged.append({
            "Workload": workload,
            "Eager PyTorch": s_row['eager_ms'],
            "PyTorch SDPA": s_row['sdpa_ms'],
            "FlashAttention-2": s_row['flash_ms'],
            "MemEff Attention": s_row['mem_eff_ms'],
            "TVM (Our Decomposed)": v_row['tvm_pipeline_loop_with_softmax_ms'],
        })
    
    return pd.DataFrame(merged)

def plot_comprehensive(df):
    # Set premium style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    
    # Data preparation
    workloads = df['Workload']
    x = np.arange(len(workloads))
    width = 0.15
    
    # Custom Colors (Premium Palette)
    colors = [
        '#607D8B', # Eager (Grey-Blue)
        '#FF9800', # SDPA (Orange)
        '#E91E63', # FA2 (Pink)
        '#9C27B0', # MemEff (Purple)
        '#00BCD4', # TVM (Cyan)
    ]
    
    columns = ["Eager PyTorch", "PyTorch SDPA", "FlashAttention-2", "MemEff Attention", "TVM (Our Decomposed)"]
    
    for i, col in enumerate(columns):
        ax.bar(x + (i - 2) * width, df[col], width, label=col, color=colors[i], zorder=3)

    # Styling
    ax.set_ylabel('Latency (ms)', fontsize=12, color='white', labelpad=10)
    ax.set_title('Comprehensive Attention Performance Benchmark\n(L40S GPU, FP16, Decomposed vs Fused)', 
                 fontsize=18, fontweight='bold', color='white', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=11, fontweight='semibold')
    
    # Log scale for better visibility of small values
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.1, zorder=0)
    
    # Legend
    legend = ax.legend(loc='upper left', frameon=True, facecolor='#121212', edgecolor='#333333', fontsize=10)
    plt.setp(legend.get_texts(), color='white')
    
    # Add annotations for speedup on the last workload (Llama-2048)
    llama_2048_idx = len(workloads) - 1
    fa2_val = df.iloc[llama_2048_idx]['FlashAttention-2']
    eager_val = df.iloc[llama_2048_idx]['Eager PyTorch']
    tvm_val = df.iloc[llama_2048_idx]['TVM (Our Decomposed)']
    
    if not np.isnan(fa2_val):
        ax.annotate(f"{eager_val/fa2_val:.1f}x vs Eager", 
                    xy=(llama_2048_idx + 0.15, fa2_val), 
                    xytext=(10, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='white'),
                    color='#E91E63', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Graph saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        data = load_and_merge()
        plot_comprehensive(data)
    except Exception as e:
        print(f"Error: {e}")
