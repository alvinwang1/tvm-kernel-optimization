import os
import modal

app = modal.App("tvm-export-kernels")
volume = modal.Volume.from_name("tvm-tuning-results")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from modal_app import image
except ModuleNotFoundError:
    image = None

def make_attn_body_mod(S, H, D, inv_scale):
    import tvm
    import tvm.te as te
    
    Q_flat = te.placeholder((S, H * D), name="Q_flat", dtype="float16")
    K_flat = te.placeholder((S, H * D), name="K_flat", dtype="float16")
    V_flat = te.placeholder((S, H * D), name="V_flat", dtype="float16")
    Out_ph = te.placeholder((H, S, D), name="Out_ph", dtype="float16")

    inv_sc = tvm.tir.const(inv_scale, "float32")

    k1 = te.reduce_axis((0, D), "k1")
    QK_raw = te.compute(
        (H, S, S),
        lambda h, i, j: te.sum(
            Q_flat[i, h * D + k1].astype("float32") * K_flat[j, h * D + k1].astype("float32"),
            axis=k1,
        ),
        name="QK_raw",
    )
    QK = te.compute((H, S, S), lambda h, i, j: QK_raw[h, i, j] * inv_sc, name="QK")

    m_ax = te.reduce_axis((0, S), "m_ax")
    RowMax = te.compute((H, S), lambda h, i: te.max(QK[h, i, m_ax], axis=m_ax), name="RowMax")

    Exp = te.compute((H, S, S), lambda h, i, j: tvm.te.exp(QK[h, i, j] - RowMax[h, i]), name="Exp")
    s_ax = te.reduce_axis((0, S), "s_ax")
    RowSum = te.compute((H, S), lambda h, i: te.sum(Exp[h, i, s_ax], axis=s_ax), name="RowSum")

    Attn = te.compute((H, S, S), lambda h, i, j: Exp[h, i, j] / RowSum[h, i], name="Attn")

    k2 = te.reduce_axis((0, S), "k2")
    Out = te.compute(
        (H, S, D),
        lambda h, i, d: te.sum(Attn[h, i, k2] * V_flat[k2, h * D + d].astype("float32"), axis=k2).astype("float16"),
        name="Out",
    )
    func = te.create_prim_func([Q_flat, K_flat, V_flat, Out])
    return tvm.IRModule({"main": func})


@app.function(image=image, volumes={"/tuning_results": volume}, timeout=3600, gpu="T4")
def export_kernels_from_modal():
    import tvm
    from tvm import meta_schedule as ms
    import glob
    
    results = {}
    arch = "sm_89"
    target = tvm.target.Target(
        f"cuda -arch={arch}"
        f" -max_num_threads=1024"
        f" -max_threads_per_block=1024"
        f" -max_shared_memory_per_block=49152"
    )

    # Search for all fused attention directories
    # They look like /tuning_results/attn_v*_attn_body_fused_*
    base_dir = "/tuning_results"
    
    if not os.path.exists(base_dir):
        return results

    work_dirs = []
    for d in os.listdir(base_dir):
        if "attn_body_fused" in d:
            work_dirs.append(os.path.join(base_dir, d))
            
    for work_dir in work_dirs:
        tag = os.path.basename(work_dir)
        tuning_record = f"{work_dir}/database_tuning_record.json"
        
        if not os.path.exists(tuning_record) or os.path.getsize(tuning_record) == 0:
            print(f"Skipping {tag}: DB does not exist or is empty.")
            continue
            
        print(f"Exporting {tag} from DB...")
        try:
            database = ms.database.JSONDatabase(work_dir=work_dir)
            workloads = database.get_all_workloads()
            if not workloads:
                print(f"  No workloads found in {tag}")
                continue
                
            # Usually just one workload per fused kernel DB
            for idx, wl in enumerate(workloads):
                records = database.get_top_k(wl, 1)
                if not records:
                    print(f"  No valid tuning records for workload {idx} in {tag}")
                    continue
                    
                best_record = records[0]
                
                # Apply the best trace to the workload's original mod
                sch = tvm.tir.Schedule(wl.mod)
                best_record.trace.apply_to_schedule(sch, remove_postproc=False)
                
                with tvm.transform.PassContext(opt_level=3):
                    lib = tvm.build(sch.mod, target=target)
                
                cuda_source = lib.imported_modules[0].get_source()
                
                # If there are multiple workloads, suffix the tag
                res_tag = tag if len(workloads) == 1 else f"{tag}_wl{idx}"
                results[res_tag] = cuda_source
                print(f"  Successfully extracted {res_tag}")
                
        except Exception as e:
            print(f"Error processing {tag}: {e}")
            
    return results

@app.local_entrypoint()
def main():
    print("Fetching tuned kernels from Modal...")
    kernels = export_kernels_from_modal.remote()
    
    if not kernels:
        print("No tuned kernels found or tuning is not yet complete.")
        return
        
    out_dir = os.path.join(os.path.dirname(__file__), "kernels", "fused")
    os.makedirs(out_dir, exist_ok=True)
    
    for tag, source in kernels.items():
        out_path = os.path.join(out_dir, f"{tag}.cu")
        with open(out_path, "w") as f:
            f.write(source)
        print(f"Saved: {out_path}")
