import modal

app = modal.App("tvm-phase1-setup")

image = (
    modal.Image
    .from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "build-essential",
        "cmake",
        "ninja-build",
        "pkg-config",
        "libtinfo-dev",
        "zlib1g-dev",
        "libxml2-dev",
        "libedit-dev",
        "libsqlite3-dev",
        "lsb-release",
        "software-properties-common",
        "gnupg",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install numpy",
    )
    .run_commands("pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio")
    .run_commands("pip install cython decorator psutil scipy tornado typing_extensions cloudpickle ml-dtypes")
    .run_commands("wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 18")
    .run_commands(
        # Pin to v0.16.0 — last stable release with meta_schedule + te + autotvm.
        # HEAD (0.24.dev) removed all legacy scheduling APIs.
        "git clone --recursive https://github.com/apache/tvm /opt/tvm",
        "cd /opt/tvm && git checkout v0.16.0 && git submodule update --init --recursive",
        "mkdir -p /opt/tvm/build",
    )
    .run_commands(
        "bash -c \"printf '%s\\n' "
        "'set(USE_CUDA ON)' "
        "'set(USE_CUDNN ON)' "
        "'set(USE_LLVM /usr/lib/llvm-18/bin/llvm-config)' "
        "'set(USE_GRAPH_EXECUTOR ON)' "
        "'set(USE_META_SCHEDULE ON)' "
        "'set(USE_RPC ON)' "
        "'set(CMAKE_BUILD_TYPE RelWithDebInfo)' "
        "> /opt/tvm/build/config.cmake\"",
    )
    .run_commands("cd /opt/tvm/build && cmake .. && cmake --build . -j$(nproc)")
    .run_commands("cd /opt/tvm && pip install -e python/")
    # ↓ New packages for SOTA benchmarking
    .run_commands("pip install xformers triton matplotlib pandas tabulate packaging")
    .env({
        "TVM_HOME": "/opt/tvm",
        "PYTHONPATH": "/opt/tvm/python:/opt/tvm:${PYTHONPATH}",
        "PATH": "/usr/lib/llvm-18/bin:${PATH}",
    })
)

@app.function(
    image=image,
    gpu="L40S",   # change to A10 or H100 if you want
    timeout=60 * 60,
)
def verify_env():
    import json
    import os
    import platform
    import subprocess
    import time

    import torch
    import tvm

    results = {}

    results["python"] = platform.python_version()
    results["platform"] = platform.platform()
    results["torch"] = torch.__version__
    results["torch_cuda"] = torch.version.cuda
    results["cuda_available"] = torch.cuda.is_available()
    results["gpu_count"] = torch.cuda.device_count()

    # Verify meta_schedule is available
    try:
        from tvm import meta_schedule as ms
        results["meta_schedule"] = "OK"
    except ImportError as e:
        results["meta_schedule"] = f"FAILED: {e}"

    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        results["device_capability"] = list(cap)

    results["tvm"] = tvm.__version__

    # nvcc version
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
        results["nvcc"] = nvcc
    except Exception as e:
        results["nvcc"] = f"unavailable: {e}"

    # nvidia-smi
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        )
        results["nvidia_smi"] = smi
    except Exception as e:
        results["nvidia_smi"] = f"unavailable: {e}"

    # simple PyTorch benchmark harness
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4096, 4096, device=device)
    y = torch.randn(4096, 4096, device=device)

    # warmup
    for _ in range(10):
        z = x @ y
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        z = x @ y
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    results["matmul_benchmark_sec"] = {
        "min": min(times),
        "max": max(times),
        "avg": sum(times) / len(times),
    }

    # TVM CUDA visibility check
    try:
        dev = tvm.cuda(0)
        results["tvm_cuda_exist"] = bool(dev.exist)
    except Exception as e:
        results["tvm_cuda_exist"] = f"error: {e}"

    return results


@app.local_entrypoint()
def main():
    out = verify_env.remote()
    print(json_pretty(out))


def json_pretty(x):
    import json
    return json.dumps(x, indent=2, sort_keys=False)