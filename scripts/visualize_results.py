import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

def load_results():
    all_data = []
    # Match results/attention_results_v6_*.json
    files = glob.glob("results/attention_results_v6_*.json")
    
    if not files:
        print("No result files found in results/")
        return None
        
    for f in files:
        # Avoid loading 'ls.json' if it's a duplicate of 'L40S.json'
        if "_ls.json" in f and os.path.exists(f.replace("_ls.json", "_L40S.json")):
            continue
            
        with open(f, "r") as fd:
            data = json.load(fd)
            # Add a simplified GPU name for plotting
            for entry in data:
                gpu_full = entry.get("gpu", "Unknown")
                if "H100" in gpu_full: entry["gpu_short"] = "H100"
                elif "A100" in gpu_full: entry["gpu_short"] = "A100"
                elif "L40S" in gpu_full: entry["gpu_short"] = "L40S"
                else: entry["gpu_short"] = entry.get("arch", "Unknown")
                
                # Calculate GFLOPS if missing (approx)
                # 2 * H * S * S * D (for QK) + 2 * H * S * S * D (for AV)
                # This is roughly 4 * H * S * S * D
                S = entry["seq_len"]
                H = entry["num_heads"]
                # Assume D=64 for BERT (12 heads) or D=128 for Llama (32 heads)
                D = 64 if H == 12 else 128
                flops = 4 * H * S * S * D
                ms = entry.get("tvm_fused_sdpa_ms", 0)
                if ms > 0:
                    entry["tvm_gflops"] = (flops / 1e9) / (ms / 1e3)
                
                pt_ms = entry.get("pt_sdpa_ms", 0)
                if pt_ms > 0:
                    entry["pt_gflops"] = (flops / 1e9) / (pt_ms / 1e3)
                    entry["tvm_v_pt_ratio"] = ms / pt_ms if pt_ms > 0 else 0

            all_data.extend(data)
    
    return pd.DataFrame(all_data)

def plot_throughput(df, out_dir):
    plt.figure(figsize=(10, 6))
    
    # Filter for Llama shapes (more compute intensive)
    llama_df = df[df["model"] == "Llama"].copy()
    llama_df = llama_df.sort_values("seq_len")
    
    for gpu in llama_df["gpu_short"].unique():
        gpu_data = llama_df[llama_df["gpu_short"] == gpu]
        plt.plot(gpu_data["seq_len"], gpu_data["tvm_gflops"], marker='o', label=f"TVM Fused ({gpu})")
        plt.plot(gpu_data["seq_len"], gpu_data["pt_gflops"], linestyle='--', alpha=0.5, label=f"PyTorch SDPA ({gpu})")

    plt.title("Attention Throughput Scaling (Llama Shapes)")
    plt.xlabel("Sequence Length")
    plt.ylabel("GFLOPS")
    plt.xscale('log', base=2)
    plt.xticks([128, 512, 2048], [128, 512, 2048])
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_scaling.png")
    print(f"Saved: {out_dir / 'throughput_scaling.png'}")

def plot_fusion_gain(df, out_dir):
    # Only pick S=2048 to show clear gains
    data_2048 = df[df["seq_len"] == 2048].copy()
    if data_2048.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Calculate unfused total
    data_2048["unfused_total"] = data_2048["tvm_qkv_ms"] + data_2048["tvm_dot_loop_ms"] + data_2048["tvm_av_loop_ms"]
    
    x = range(len(data_2048))
    width = 0.35
    
    plt.bar(x, data_2048["unfused_total"], width, label='Unfused Pipeline (Sum of Kernels)', color='gray', alpha=0.5)
    plt.bar([i + width for i in x], data_2048["tvm_fused_sdpa_ms"], width, label='TVM Fused Kernel', color='blue')
    
    plt.ylabel('Latency (ms)')
    plt.title('Fusion Impact: Component Pipeline vs. Fused Kernel (S=2048)')
    plt.xticks([i + width/2 for i in x], data_2048["gpu_short"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fusion_impact.png")
    print(f"Saved: {out_dir / 'fusion_impact.png'}")

def main():
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    
    df = load_results()
    if df is None or df.empty:
        print("No data to plot.")
        return
        
    print(f"Loaded {len(df)} benchmark entries.")
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-muted')
    except:
        pass
        
    plot_throughput(df, out_dir)
    plot_fusion_gain(df, out_dir)
    
    print("\nVisualization complete. Check the 'plots/' directory.")

if __name__ == "__main__":
    main()
