import os
import modal
import re
import shutil

app = modal.App("tvm-export-kernels-v6")
volume = modal.Volume.from_name("tvm-tuning-results")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from modal_app import image
except ModuleNotFoundError:
    image = None

@app.function(image=image, volumes={"/tuning_results": volume}, timeout=3600, gpu="T4")
def export_kernels_to_volume():
    import tvm
    from tvm import meta_schedule as ms
    import tvm.contrib.nvcc  # noqa
    try:
        from tvm.tir.tensor_intrin import cuda as _  # noqa
    except ImportError:
        pass
    
    base_dir = "/tuning_results"
    out_dir = "/tuning_results/exported_kernels"
    
    # Clear previous exports
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    work_dirs = []
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "database_tuning_record.json")):
            if "_v6_" in d or "_v4_" in d:
                work_dirs.append(path)
            
    if not work_dirs:
        print("No v4/v6 tuning databases found in volume.")
        return

    print(f"Found {len(work_dirs)} databases. Starting export...")

    for work_dir in work_dirs:
        tag = os.path.basename(work_dir)
        tuning_record = f"{work_dir}/database_tuning_record.json"
        
        if os.path.getsize(tuning_record) == 0:
            continue
            
        arch_match = re.search(r"sm_(\d+)", tag)
        arch = arch_match.group(0) if arch_match else "sm_89"
        
        target = tvm.target.Target(f"cuda -arch={arch}")

        try:
            database = ms.database.JSONDatabase(work_dir=work_dir)
            records = database.get_all_tuning_records()
            if not records:
                continue
                
            workload_to_best_record = {}
            for record in records:
                wl = record.workload
                avg_time = sum(record.run_secs) / len(record.run_secs) if record.run_secs else float('inf')
                if wl not in workload_to_best_record or avg_time < workload_to_best_record[wl][0]:
                    workload_to_best_record[wl] = (avg_time, record)
            
            for idx, (wl, (avg_time, best_record)) in enumerate(workload_to_best_record.items()):
                sch = tvm.tir.Schedule(wl.mod)
                best_record.trace.apply_to_schedule(sch, remove_postproc=False)
                with tvm.transform.PassContext(opt_level=3):
                    lib = tvm.build(sch.mod, target=target)
                
                cuda_source = str(lib.imported_modules[0].get_source())
                
                # Determine category
                if "attn_body_fused" in tag: category = "fused_v4"
                elif "qk_gemm_v6" in tag: category = "fused_v6_qk"
                elif "softmax_av_v6" in tag: category = "fused_v6_sav"
                elif any(x in tag for x in ["Q_proj", "K_proj", "V_proj"]): category = "gemm_qkv"
                elif any(x in tag for x in ["QK_dot", "AV_sum"]): category = "gemm_attn"
                else: category = "misc"
                
                res_tag = tag if len(workload_to_best_record) == 1 else f"{tag}_wl{idx}"
                dest_dir = os.path.join(out_dir, category)
                os.makedirs(dest_dir, exist_ok=True)
                
                with open(os.path.join(dest_dir, f"{res_tag}.cu"), "w") as f:
                    f.write(cuda_source)
                print(f"  Exported {res_tag} to volume")
                
        except Exception as e:
            print(f"Error processing {tag}: {e}")
            
    volume.commit()
    print("Export complete. Committed to volume.")

@app.local_entrypoint()
def main():
    export_kernels_to_volume.remote()
    print("\nKernels have been exported to the Modal volume.")
    print("To pull them to your local machine, run:")
    print("modal volume get tvm-tuning-results exported_kernels scripts/kernels")
