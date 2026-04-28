# TVM Kernel Optimization for Attention Mechanisms

A systematic exploration of optimizing attention mechanism components using **Apache TVM (MetaSchedule)**. This repository focuses on decomposing the standard Scaled Dot-Product Attention (SDPA) into its constituent kernels (Projections, QK Dot, AV Sum) and optimizing each for specific hardware targets (NVIDIA L40S).

## 🚀 Overview

The project aims to answer whether decomposed, auto-tuned kernels can compete with highly optimized fused implementations like **FlashAttention-2** or **PyTorch SDPA**.

### Key Components Optimized:
- **QKV Projections**: Input linear transformations.
- **QK Dot Product**: Score calculation (Query × Key).
- **AV Summation**: Weighted value aggregation (Attention × Value).

## 📂 Repository Structure

```text
.
├── analysis/               # Scripts for results processing and plot generation
├── benchmarks/             # Core benchmarking and tuning logic
│   ├── legacy/             # Archived versions of early experiments
│   ├── tune_attention.py   # Main entry point for attention component tuning
│   └── benchmark_sota_attention.py # Benchmarking against SOTA (FA2, SDPA, etc.)
├── kernels/                # TVM-generated CUDA kernels (.cu)
├── reports/                # Performance reports and visualizations (.png, .md)
├── results/                # Raw benchmark data (JSON)
├── scripts/                # Misc utility and setup scripts
├── utils/                  # TIR inspection and helper utilities
└── modal_app.py            # Modal environment configuration and image definition
```

## 🛠️ Setup & Usage

This project uses [Modal](https://modal.com/) to manage GPU environments and scaling.

### Prerequisites
- Python 3.11+
- Modal token (`modal token new`)

### Running the Benchmarks

1.  **Tune and Benchmark Attention Components**:
    ```bash
    modal run benchmarks/tune_attention.py --max-trials 800
    ```

2.  **Compare against SOTA (FlashAttention-2, etc.)**:
    ```bash
    modal run benchmarks/benchmark_sota_attention.py
    ```

3.  **Generate Comprehensive Plots**:
    ```bash
    python analysis/comprehensive_comparison.py
    ```

## 📊 Performance Insights

We compare our TVM-optimized kernels against:
- **Eager PyTorch**: Baseline matmul.
- **PyTorch SDPA**: Standard library fused attention.
- **FlashAttention-2**: State-of-the-art fused kernel.
- **Memory-Efficient Attention (xFormers)**: Cutlass-based implementation.

Visualizations and detailed reports can be found in the `reports/` directory.

## 📜 License
MIT
