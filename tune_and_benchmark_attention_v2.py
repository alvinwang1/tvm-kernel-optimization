"""Compatibility entrypoint for the v2 attention benchmark.

This forwards to the canonical implementation in:
    benchmarks/tune_and_benchmark_attention_v2.py
"""

from benchmarks.tune_and_benchmark_attention_v2 import app, main, tune_and_benchmark_attention_v2

__all__ = ["app", "main", "tune_and_benchmark_attention_v2"]

if __name__ == "__main__":
    main()
