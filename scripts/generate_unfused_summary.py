import json
import glob
import os

def generate_unfused_summary():
    results = []
    # Match results/attention_results_v6_*.json
    files = glob.glob("results/attention_results_v6_*.json")
    
    for f in files:
        if "_ls.json" in f: continue
        # Extract GPU name from filename
        gpu = f.split("_")[-1].replace(".json", "")
        
        with open(f, "r") as fd:
            data = json.load(fd)
            for entry in data:
                pt_eager = entry.get("pt_full_unfused_batched_ms")
                tvm_pipe = entry.get("tvm_pipeline_loop_with_softmax_ms")
                
                if pt_eager and tvm_pipe:
                    results.append({
                        "gpu": gpu,
                        "workload": f"{entry['model']}-{entry['seq_len']}",
                        "pytorch_eager_ms": pt_eager,
                        "tvm_component_pipeline_ms": tvm_pipe,
                        "speedup_tvm_vs_pt": pt_eager / tvm_pipe
                    })

    with open("results/unfused_comparison_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Generated results/unfused_comparison_summary.json")

if __name__ == "__main__":
    generate_unfused_summary()
