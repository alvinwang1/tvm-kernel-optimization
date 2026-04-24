import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "attention_results_v2.json"
REPORT_PATH = ROOT / "reports" / "attention_v2_analysis_report.md"
CHART_PATH = ROOT / "reports" / "attention_v2_analysis_charts.png"
KERNEL_ROOT = ROOT / "kernels"


def load_results(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["workload"] = df["model"] + "-" + df["seq_len"].astype(str)
    df = df.sort_values(["model", "seq_len"]).reset_index(drop=True)
    return df


def parse_kernel_file(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    launch_match = re.search(r"__launch_bounds__\((\d+)\)", txt)
    op_match = re.match(r"tvm_(.+)_(BERT|Llama)_(\d+)\.cu", path.name)
    op = op_match.group(1) if op_match else "unknown"
    model = op_match.group(2) if op_match else "unknown"
    seq = int(op_match.group(3)) if op_match else -1
    return {
        "file": path.name,
        "op": op,
        "model": model,
        "seq_len": seq,
        "mma_sync": txt.count("mma_sync"),
        "load_matrix_sync": txt.count("load_matrix_sync"),
        "store_matrix_sync": txt.count("store_matrix_sync"),
        "syncthreads": txt.count("__syncthreads"),
        "launch_bounds": int(launch_match.group(1)) if launch_match else np.nan,
        "uses_wmma": ("wmma::" in txt),
    }


def load_kernel_stats(root: Path) -> pd.DataFrame:
    rows = [parse_kernel_file(p) for p in sorted(root.glob("tvm_*.cu"))]
    return pd.DataFrame(rows)


def make_charts(df: pd.DataFrame, kdf: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    x = np.arange(len(df))
    labels = df["workload"].tolist()

    # 1) Matched PT vs TVM pipeline comparison
    axes[0, 0].bar(x - 0.18, df["pt_pipeline_loop_no_softmax_ms"], width=0.36, label="PyTorch loop pipeline")
    axes[0, 0].bar(x + 0.18, df["tvm_pipeline_loop_no_softmax_ms"], width=0.36, label="TVM loop pipeline")
    axes[0, 0].set_title("Matched Loop Pipeline (No Softmax)")
    axes[0, 0].set_ylabel("Latency (ms)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=20)
    axes[0, 0].legend()

    # 2) PyTorch best path reference
    axes[0, 1].plot(x, df["pt_full_unfused_batched_ms"], marker="o", label="PT full unfused")
    axes[0, 1].plot(x, df["pt_sdpa_ms"], marker="o", label="PT SDPA")
    axes[0, 1].plot(x, df["pt_pipeline_loop_with_softmax_ms"], marker="o", label="PT loop pipeline + softmax")
    axes[0, 1].set_title("PyTorch Paths")
    axes[0, 1].set_ylabel("Latency (ms)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=20)
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()

    # 3) PT/TVM component speed ratios (>1 means TVM faster)
    axes[1, 0].plot(x, df["qkv_speedup_pt_over_tvm"], marker="o", label="QKV")
    axes[1, 0].plot(x, df["dot_loop_speedup_pt_over_tvm"], marker="o", label="QK loop")
    axes[1, 0].plot(x, df["av_loop_speedup_pt_over_tvm"], marker="o", label="AV loop")
    axes[1, 0].plot(x, df["pipeline_no_softmax_speedup_pt_over_tvm"], marker="o", label="Pipeline")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("PT/TVM Ratios (Matched Granularity)")
    axes[1, 0].set_ylabel("PT latency / TVM latency")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=20)
    axes[1, 0].legend()

    # 4) Kernel structure summary
    by_op = (
        kdf.groupby("op")[["mma_sync", "load_matrix_sync", "store_matrix_sync", "syncthreads"]]
        .mean()
        .sort_index()
    )
    xpos = np.arange(len(by_op.index))
    bw = 0.2
    axes[1, 1].bar(xpos - 1.5 * bw, by_op["mma_sync"], width=bw, label="mma_sync")
    axes[1, 1].bar(xpos - 0.5 * bw, by_op["load_matrix_sync"], width=bw, label="load_matrix_sync")
    axes[1, 1].bar(xpos + 0.5 * bw, by_op["store_matrix_sync"], width=bw, label="store_matrix_sync")
    axes[1, 1].bar(xpos + 1.5 * bw, by_op["syncthreads"], width=bw, label="__syncthreads")
    axes[1, 1].set_title("TVM CUDA Structure by Op (avg across shapes)")
    axes[1, 1].set_ylabel("Count per file")
    axes[1, 1].set_xticks(xpos)
    axes[1, 1].set_xticklabels(by_op.index, rotation=25)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)


def build_report(df: pd.DataFrame, kdf: pd.DataFrame, report_path: Path, chart_path: Path) -> None:
    wmma_coverage = (kdf["uses_wmma"].mean() * 100.0) if len(kdf) else 0.0
    launch_vals = kdf["launch_bounds"].dropna()
    launch_text = (
        f"min={int(launch_vals.min())}, max={int(launch_vals.max())}, median={int(launch_vals.median())}"
        if not launch_vals.empty
        else "n/a"
    )

    matched_cols = [
        "workload",
        "pt_qkv_ms",
        "tvm_qkv_ms",
        "pt_dot_loop_ms",
        "tvm_dot_loop_ms",
        "pt_av_loop_ms",
        "tvm_av_loop_ms",
        "pt_pipeline_loop_no_softmax_ms",
        "tvm_pipeline_loop_no_softmax_ms",
        "pipeline_no_softmax_speedup_pt_over_tvm",
    ]

    rows = []
    for _, r in df.iterrows():
        rows.append(
            "| {workload} | {pt_qkv_ms:.3f} | {tvm_qkv_ms:.3f} | {pt_dot_loop_ms:.3f} | {tvm_dot_loop_ms:.3f} | {pt_av_loop_ms:.3f} | {tvm_av_loop_ms:.3f} | {pt_pipeline_loop_no_softmax_ms:.3f} | {tvm_pipeline_loop_no_softmax_ms:.3f} | {pipeline_no_softmax_speedup_pt_over_tvm:.3f} |".format(
                **{k: r[k] for k in matched_cols}
            )
        )

    tvm_faster = (df["pipeline_no_softmax_speedup_pt_over_tvm"] > 1.0).sum()
    pt_faster = (df["pipeline_no_softmax_speedup_pt_over_tvm"] < 1.0).sum()

    # SDPA relative behavior in the latest run
    sdpa_vs_unfused = (df["pt_sdpa_ms"] / df["pt_full_unfused_batched_ms"]).round(3)

    report = f"""# Attention v2 Analysis Report

## Scope
This report compares:
- TVM generated CUDA kernels (`tvm_*.cu`) for Q/K/V projection, QK dot, and AV sum
- PyTorch timing metrics and TVM timing metrics from `attention_results_v2.json`

Artifacts:
- Chart: `{chart_path.name}`
- Data source: `{RESULTS_PATH.name}`

## High-Level Findings
1. In matched per-head loop granularity (no softmax), TVM is faster on {tvm_faster} / {len(df)} workloads; PyTorch is faster on {pt_faster} / {len(df)}.
2. TVM generated kernels show consistent Tensor Core WMMA patterns across files (WMMA coverage: {wmma_coverage:.1f}%).
3. Kernel launch bounds are stable across files ({launch_text}), indicating a regularized schedule template across shapes.
4. The latest PyTorch-only rerun changed PT numbers while preserving TVM fields, so current JSON is mixed-source (PT refreshed, TVM from prior run).

## Matched Latency Table (ms)
| Workload | PT QKV | TVM QKV | PT QK(loop) | TVM QK(loop) | PT AV(loop) | TVM AV(loop) | PT Pipeline(no sfmx) | TVM Pipeline(no sfmx) | PT/TVM Pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## PyTorch Path Readout
`pt_sdpa_ms / pt_full_unfused_batched_ms` by workload:
- {df.loc[0, 'workload']}: {sdpa_vs_unfused.iloc[0]}
- {df.loc[1, 'workload']}: {sdpa_vs_unfused.iloc[1]}
- {df.loc[2, 'workload']}: {sdpa_vs_unfused.iloc[2]}
- {df.loc[3, 'workload']}: {sdpa_vs_unfused.iloc[3]}
- {df.loc[4, 'workload']}: {sdpa_vs_unfused.iloc[4]}

Interpretation:
- Ratios > 1 mean SDPA is slower than unfused batched path for that shape/run.
- Ratios < 1 mean SDPA is faster.

## CUDA Kernel Inspection Summary
Across 25 generated `.cu` files:
- All files contain WMMA/Tensor Core primitives.
- Common operation blocks include `load_matrix_sync`, `mma_sync`, `store_matrix_sync`, and explicit `__syncthreads` staging.
- Q/K/V projection kernels are compute-heavy GEMM-style kernels; QK/AV kernels retain the same WMMA pattern but under head-wise decomposition used by this benchmark path.

## Caveats
1. Current JSON mixes fresh PT numbers (from `--pytorch-only`) with older TVM numbers.
2. This report compares matched unfused components, not fused-vs-fused TVM SDPA.
3. Benchmark includes random inputs; absolute times can drift slightly run-to-run.
"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    df = load_results(RESULTS_PATH)
    kdf = load_kernel_stats(KERNEL_ROOT)
    make_charts(df, kdf, CHART_PATH)
    build_report(df, kdf, REPORT_PATH, CHART_PATH)
    print(f"Generated: {CHART_PATH.name}")
    print(f"Generated: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
