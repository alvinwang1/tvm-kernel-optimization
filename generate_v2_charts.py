"""
generate_v2_charts.py
=====================
Generates analysis charts for the v2 TVM vs PyTorch benchmark.
Run locally:  python generate_v2_charts.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Data ─────────────────────────────────────────────────────────────────────
with open("attention_results_v2.json") as f:
    data = json.load(f)

# Build a label for each shape
def label(r):
    return f"{r['model']}\nseq={r['seq_len']}"

labels     = [label(r) for r in data]
short_lbls = [f"{r['model']}-{r['seq_len']}" for r in data]

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (colour-blind friendly)
C_PT  = "#4C72B0"   # blue  – PyTorch
C_TVM = "#DD8452"   # orange – TVM
C_SDPA = "#55A868"  # green  – SDPA
C_UNF = "#C44E52"   # red    – unfused PT batched (reference)
ACCENT = "#8172B2"  # purple – speedup line

# ─────────────────────────────────────────────────────────────────────────────
# ── Figure 1: raw latency + speedup ratios ───────────────────────────────────
fig1 = plt.figure(figsize=(20, 16))
fig1.patch.set_facecolor("#F8F8F8")
gs1 = GridSpec(2, 2, figure=fig1, hspace=0.42, wspace=0.30)

ax1 = fig1.add_subplot(gs1[0, :])   # full-width grouped bar
ax2 = fig1.add_subplot(gs1[1, 0])   # speedup bars
ax3 = fig1.add_subplot(gs1[1, 1])   # pipeline comparison

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor("#FAFAFA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")

# ── Figure 2: fusion gap + scaling line ──────────────────────────────────────
fig2 = plt.figure(figsize=(18, 8))
fig2.patch.set_facecolor("#F8F8F8")
gs2 = GridSpec(1, 2, figure=fig2, hspace=0.30, wspace=0.32)

ax4 = fig2.add_subplot(gs2[0, 0])   # SDPA fusion gap
ax5 = fig2.add_subplot(gs2[0, 1])   # speedup vs seq_len

for ax in [ax4, ax5]:
    ax.set_facecolor("#FAFAFA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Grouped bar: PT loop vs TVM per operation
# ══════════════════════════════════════════════════════════════════════════════
n   = len(data)
x   = np.arange(n)
w   = 0.15   # bar width
ops = [
    ("pt_qkv_ms",       "tvm_qkv_ms",       "QKV proj"),
    ("pt_dot_loop_ms",  "tvm_dot_loop_ms",  "QK dot (loop)"),
    ("pt_av_loop_ms",   "tvm_av_loop_ms",   "AV sum (loop)"),
]
offsets = np.array([-2, 0, 2]) * w          # centre the 3 pairs

for i, (pt_key, tvm_key, op_name) in enumerate(ops):
    pt_vals  = [r[pt_key]  for r in data]
    tvm_vals = [r[tvm_key] for r in data]
    off = offsets[i]
    b1 = ax1.bar(x + off - w/2, pt_vals,  w, color=C_PT,  alpha=0.85, label=None)
    b2 = ax1.bar(x + off + w/2, tvm_vals, w, color=C_TVM, alpha=0.85, label=None)

    # Op labels above each pair
    for xi, (pv, tv) in enumerate(zip(pt_vals, tvm_vals)):
        top = max(pv, tv) + 0.01
        ax1.text(x[xi] + off, top, op_name, ha="center", va="bottom",
                 fontsize=6.5, color="#444", rotation=90)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Latency (ms)", fontsize=10)
ax1.set_title("Operation-Level Latency: PyTorch Loop vs TVM (per-head loop, lower is better)",
              fontsize=12, fontweight="bold", pad=10)
pt_patch  = mpatches.Patch(color=C_PT,  alpha=0.85, label="PyTorch (cuBLAS / loop)")
tvm_patch = mpatches.Patch(color=C_TVM, alpha=0.85, label="TVM MetaSchedule (WMMA)")
ax1.legend(handles=[pt_patch, tvm_patch], fontsize=9, loc="upper left")
ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax1.set_ylim(bottom=0)

# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Speedup bars (PT latency / TVM latency) for each op × shape
# ══════════════════════════════════════════════════════════════════════════════
speedup_ops = [
    ("qkv_speedup_pt_over_tvm",         "QKV proj"),
    ("dot_loop_speedup_pt_over_tvm",    "QK dot"),
    ("av_loop_speedup_pt_over_tvm",     "AV sum"),
    ("pipeline_no_softmax_speedup_pt_over_tvm", "Full pipeline"),
]
sn = len(speedup_ops)
sw = 0.15
sx = np.arange(n)
s_offsets = np.linspace(-(sn-1)/2, (sn-1)/2, sn) * sw

colors_sp = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, (key, name) in enumerate(speedup_ops):
    vals = [r[key] for r in data]
    bars = ax2.bar(sx + s_offsets[i], vals, sw, color=colors_sp[i], alpha=0.83, label=name)

ax2.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Parity (1.0×)")
ax2.set_xticks(sx)
ax2.set_xticklabels(short_lbls, fontsize=8, rotation=15, ha="right")
ax2.set_ylabel("PT latency / TVM latency  (>1 = TVM faster)", fontsize=9)
ax2.set_title("Speedup Ratios by Operation\n(>1.0 means TVM is faster)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=7.5, loc="upper left", ncol=2)
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax2.set_ylim(bottom=0)

# Annotate parity crossings
for xi, r in enumerate(data):
    qkv_sp = r["qkv_speedup_pt_over_tvm"]
    if qkv_sp < 1.0:
        ax2.annotate("TVM\nslower", xy=(sx[xi] + s_offsets[0], qkv_sp),
                     xytext=(sx[xi] + s_offsets[0], qkv_sp - 0.12),
                     fontsize=6, ha="center", color="#4C72B0",
                     arrowprops=dict(arrowstyle="-", color="#4C72B0", lw=0.7))

# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Pipeline comparison: PT loop no-softmax vs TVM vs SDPA
# ══════════════════════════════════════════════════════════════════════════════
pt_pipe  = [r["pt_pipeline_loop_no_softmax_ms"]  for r in data]
tvm_pipe = [r["tvm_pipeline_loop_no_softmax_ms"] for r in data]
sdpa     = [r["pt_sdpa_ms"]                      for r in data]
unfused_batched = [r["pt_full_unfused_batched_ms"] for r in data]

pw = 0.18
px = np.arange(n)
ax3.bar(px - 1.5*pw, pt_pipe,         pw, color=C_PT,   alpha=0.85, label="PT loop pipeline (no softmax)")
ax3.bar(px - 0.5*pw, tvm_pipe,        pw, color=C_TVM,  alpha=0.85, label="TVM pipeline (no softmax)")
ax3.bar(px + 0.5*pw, unfused_batched, pw, color=C_UNF,  alpha=0.85, label="PT unfused batched")
ax3.bar(px + 1.5*pw, sdpa,            pw, color=C_SDPA, alpha=0.85, label="PT SDPA (Flash Attn)")

ax3.set_xticks(px)
ax3.set_xticklabels(short_lbls, fontsize=8, rotation=15, ha="right")
ax3.set_ylabel("End-to-End Latency (ms)", fontsize=9)
ax3.set_title("Full Pipeline Latency Comparison\n(SDPA is fused — different algorithm)", fontsize=11, fontweight="bold")
ax3.legend(fontsize=7.5, loc="upper left")
ax3.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax3.set_ylim(bottom=0)

# ══════════════════════════════════════════════════════════════════════════════
# Chart 4 — SDPA overhead factor vs TVM pipeline (shows fusion gap)
# ══════════════════════════════════════════════════════════════════════════════
tvm_over_sdpa = [r["tvm_pipeline_loop_no_softmax_ms"] / r["pt_sdpa_ms"] for r in data]
pt_over_sdpa  = [r["pt_pipeline_loop_no_softmax_ms"] / r["pt_sdpa_ms"]  for r in data]

ax4.plot(short_lbls, pt_over_sdpa,  "o--", color=C_PT,  linewidth=1.8, markersize=7,
         label="PT loop pipeline / SDPA")
ax4.plot(short_lbls, tvm_over_sdpa, "s-",  color=C_TVM, linewidth=1.8, markersize=7,
         label="TVM pipeline / SDPA")
ax4.axhline(1.0, color="grey", linewidth=1, linestyle=":", label="Parity with SDPA")
ax4.fill_between(short_lbls, tvm_over_sdpa, 1.0, alpha=0.10, color=C_TVM)
ax4.set_ylabel("Unfused cost / SDPA cost  (×)", fontsize=9)
ax4.set_title("Memory Fusion Gap:\nHow Many × Slower Than SDPA?", fontsize=11, fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax4.tick_params(axis="x", labelsize=8)
for xi, (tv, pv) in enumerate(zip(tvm_over_sdpa, pt_over_sdpa)):
    ax4.text(xi, tv + 0.08, f"{tv:.1f}×", ha="center", fontsize=7.5, color=C_TVM, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 5 — How speedup evolves with seq_len (Llama shapes only)
# ══════════════════════════════════════════════════════════════════════════════
llama = [r for r in data if r["model"] == "Llama"]
seqs  = [r["seq_len"] for r in llama]

qkv_sp  = [r["qkv_speedup_pt_over_tvm"]         for r in llama]
qk_sp   = [r["dot_loop_speedup_pt_over_tvm"]     for r in llama]
av_sp   = [r["av_loop_speedup_pt_over_tvm"]      for r in llama]
pipe_sp = [r["pipeline_no_softmax_speedup_pt_over_tvm"] for r in llama]

ax5.plot(seqs, qkv_sp,  "o-",  color="#4C72B0", linewidth=2, markersize=8, label="QKV projection")
ax5.plot(seqs, qk_sp,   "s-",  color="#DD8452", linewidth=2, markersize=8, label="QK dot (loop)")
ax5.plot(seqs, av_sp,   "^-",  color="#55A868", linewidth=2, markersize=8, label="AV sum (loop)")
ax5.plot(seqs, pipe_sp, "D--", color="#8172B2", linewidth=2, markersize=8, label="Full pipeline")
ax5.axhline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
ax5.fill_between(seqs, 1.0, [min(v, 1.0) for v in qkv_sp], alpha=0.08, color="#4C72B0",
                 label="TVM slower region")

ax5.set_xscale("log")
ax5.set_xticks(seqs)
ax5.set_xticklabels([str(s) for s in seqs], fontsize=9)
ax5.set_xlabel("Sequence Length (log scale)  — Llama-7B shapes", fontsize=9)
ax5.set_ylabel("PT latency / TVM latency  (>1 = TVM faster)", fontsize=9)
ax5.set_title("How Speedup Scales with Sequence Length\n(Llama-7B, hidden=4096, 32 heads)",
              fontsize=11, fontweight="bold")
ax5.legend(fontsize=8.5)
ax5.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)

# Value labels
for s, qk, av in zip(seqs, qk_sp, av_sp):
    ax5.text(s, qk + 0.04, f"{qk:.2f}×", ha="center", fontsize=7.5, color="#DD8452")
    ax5.text(s, av - 0.10, f"{av:.2f}×", ha="center", fontsize=7.5, color="#55A868")

# ─────────────────────────────────────────────────────────────────────────────
fig1.suptitle(
    "TVM MetaSchedule vs PyTorch/cuBLAS — Attention Benchmark v2  [Part 1 of 2]\n"
    "GPU: NVIDIA L40S (sm_89)  |  FP16  |  800 MetaSchedule trials per shape",
    fontsize=13, fontweight="bold", y=1.01
)
out1 = "analysis_charts_v2_part1.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
print(f"Saved: {out1}")
plt.close(fig1)

fig2.suptitle(
    "TVM MetaSchedule vs PyTorch/cuBLAS — Attention Benchmark v2  [Part 2 of 2]\n"
    "Fusion Gap vs Flash Attention  |  Speedup Scaling with Sequence Length",
    fontsize=13, fontweight="bold", y=1.04
)
out2 = "analysis_charts_v2_part2.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Saved: {out2}")
plt.close(fig2)
