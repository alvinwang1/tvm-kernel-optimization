# Attention v2 Analysis Report

## Scope
This report compares:
- TVM generated CUDA kernels (`tvm_*.cu`) for Q/K/V projection, QK dot, and AV sum
- PyTorch timing metrics and TVM timing metrics from `attention_results_v2.json`

Artifacts:
- Chart: `attention_v2_analysis_charts.png`
- Data source: `attention_results_v2.json`

## High-Level Findings
1. In matched per-head loop granularity (no softmax), TVM is faster on 4 / 5 workloads; PyTorch is faster on 1 / 5.
2. TVM generated kernels show consistent Tensor Core WMMA patterns across files (WMMA coverage: 100.0%).
3. Kernel launch bounds are stable across files (min=32, max=512, median=128), indicating a regularized schedule template across shapes.


## Matched Latency Table (ms)
| Workload | PT QKV | TVM QKV | PT QK(loop) | TVM QK(loop) | PT AV(loop) | TVM AV(loop) | PT Pipeline(no sfmx) | TVM Pipeline(no sfmx) | PT/TVM Pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT-128 | 0.029 | 0.030 | 0.233 | 0.092 | 0.176 | 0.092 | 0.437 | 0.214 | 1.209 |
| BERT-512 | 0.026 | 0.037 | 0.253 | 0.111 | 0.175 | 0.110 | 0.454 | 0.258 | 1.189 |
| Llama-128 | 0.087 | 0.131 | 0.533 | 0.240 | 0.468 | 0.238 | 1.087 | 0.608 | 1.262 |
| Llama-512 | 0.244 | 0.327 | 0.541 | 0.298 | 0.445 | 0.307 | 1.230 | 0.933 | 1.390 |
| Llama-2048 | 0.959 | 0.958 | 0.529 | 0.469 | 0.649 | 0.684 | 2.137 | 2.112 | 0.963 |

## PyTorch Path Readout
`pt_sdpa_ms / pt_full_unfused_batched_ms` by workload:
- BERT-128: 1.138
- BERT-512: 1.279
- Llama-128: 0.854
- Llama-512: 0.914
- Llama-2048: 0.387

Interpretation:
- Ratios > 1 mean SDPA is slower than unfused batched path for that shape/run.
- Ratios < 1 mean SDPA is faster.

## CUDA Kernel Inspection Summary
Across 25 generated `.cu` files:
- All files contain WMMA/Tensor Core primitives.
- Common operation blocks include `load_matrix_sync`, `mma_sync`, `store_matrix_sync`, and explicit `__syncthreads` staging.
- Q/K/V projection kernels are compute-heavy GEMM-style kernels; QK/AV kernels retain the same WMMA pattern but under head-wise decomposition used by this benchmark path.

## Caveats
1. This report compares matched unfused components, not fused-vs-fused TVM SDPA.
2. Benchmark includes random inputs; absolute times can drift slightly run-to-run.
