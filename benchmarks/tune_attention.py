"""
tune_attention.py
===================================
Fairness-focused benchmark for TVM vs PyTorch attention.

What this script benchmarks:
  Component pipeline (existing):
    - Separate tuned TVM GEMM kernels: Q/K/V projection, QK dot (per-head
      loop), AV sum (per-head loop).  Compared against matched PyTorch loop
      and batched baselines.

  Fused SDPA (v6 — corrected layout):
    Four kernels in sequence:

    (a) QKV projection  – existing tuned GEMMs (cached under attn_v4_ prefix).
        Output: Q_flat, K_flat, V_flat  shape (S, H*D).

    (b) Reshape kernel  – trivial elementwise copy reindexing
            (S, H*D) -> (H, S, D)
        Runs for Q, K, V.  Always uses a simple hand-written schedule
        (purely bandwidth-bound and tiny — no need to tune).

    (c) QK GEMM  – clean batched GEMM, MetaSchedule-friendly:
            QK[h,i,j] = inv_scale * sum_d  Q[h,i,d] * K[h,j,d]
        Input shape: Q[H,S,D], K[H,S,D].
        MultiLevelTiling fires because the reduce axis (d) maps cleanly
        to the last dimension of Q and K — no strided h*D+d indexing.
        (Previous versions used Q_flat[i, h*D + k1] which combines a
        spatial loop with the reduce axis, breaking pattern recognition.)

    (d) SoftmaxAV kernel  – chained reductions + batched GEMM:
            RowMax[h,i]  = max_j QK[h,i,j]
            Exp[h,i,j]   = exp(QK[h,i,j] - RowMax[h,i])
            RowSum[h,i]  = sum_j Exp[h,i,j]
            Attn[h,i,j]  = Exp[h,i,j] / RowSum[h,i]
            Out[h,i,d]   = sum_j Attn[h,i,j] * V[h,j,d]
        Input: QK[H,S,S], V[H,S,D] — V also clean (no strided access).
        CrossThreadReduction excluded from space generator (crashes on
        RowMax/RowSum before threadIdx.x is bound).
        Fallback uses compute_at to pull RowMax/RowSum inside Out tile
        (no inter-block race -> no NaN).

  Cache compatibility:
    GEMM kernels (Q/K/V proj, QK_dot, AV_sum) continue to use the
    attn_v4_ work_dir prefix so existing tuned caches are reused.
    New fused kernels (QK GEMM, SoftmaxAV) use attn_v6_ prefix.

  Limitations:
    * Materialises the full (H x S x S) attention matrix — not FlashAttention.
    * PyTorch SDPA may use FlashAttention-2; latency comparison is
      informational, not apples-to-apples in kernel algorithm.

Usage:
    modal run benchmarks/tune_attention.py
    modal run benchmarks/tune_attention.py --max-trials 800 --iters 200
    modal run benchmarks/tune_attention.py --max-trials 800 --fused-only
"""

import modal
from pathlib import Path

try:
    from modal_app import image
except ModuleNotFoundError:
    image = None

app = modal.App("tvm-attention-benchmark-v6")

volume = modal.Volume.from_name("tvm-tuning-results", create_if_missing=True)

# (seq_len, hidden_dim, num_heads, model_name)
ATTENTION_SHAPES = [
    (128,  768,  12, "BERT"),
    (512,  768,  12, "BERT"),
    (128,  4096, 32, "Llama"),
    (512,  4096, 32, "Llama"),
    (2048, 4096, 32, "Llama"),
]


def _timeit(fn, iters=200, warmup=20):
    import time
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


import os
gpu_type = os.environ.get("MODAL_GPU", "L40S")

@app.function(
    image=image,
    gpu=gpu_type,
    timeout=86400,
    volumes={"/tuning_results": volume},
)
def tune_and_benchmark_attention_v6(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    model: str,
    max_trials: int = 800,
    iters: int = 200,
    pytorch_only: bool = False,
    fused_only: bool = False,
):
    import math
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tvm
    import tvm.te as te
    import tvm.contrib.nvcc  # noqa – registers wmma TensorIntrins
    from tvm import meta_schedule as ms
    try:
        from tvm.tir.tensor_intrin import cuda as _  # noqa
    except ImportError:
        pass

    assert torch.cuda.is_available(), "No CUDA device found"

    gpu_name = torch.cuda.get_device_name(0)
    cap      = torch.cuda.get_device_capability(0)
    arch     = f"sm_{cap[0]}{cap[1]}"
    dev      = tvm.cuda(0)

    target = tvm.target.Target(
        f"cuda -arch={arch}"
        f" -max_num_threads=1024"
        f" -max_threads_per_block=1024"
        f" -max_shared_memory_per_block=49152"
    )

    head_dim  = hidden_dim // num_heads
    scale     = math.sqrt(head_dim)
    inv_scale = float(1.0 / scale)

    print(f"\n{'=' * 68}")
    print(f"[v6] {model} seq={seq_len} hidden={hidden_dim} heads={num_heads} "
          f"on {gpu_name} ({arch})")
    print(f"{'=' * 68}")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    X  = torch.randn(seq_len,    hidden_dim, dtype=torch.float16, device="cuda")
    Wq = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")
    Wk = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")
    Wv = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device="cuda")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_tvm(t: torch.Tensor):
        return tvm.nd.array(t.detach().cpu().numpy(), dev)

    def zeros_tvm(shape, dtype="float16"):
        return tvm.nd.array(np.zeros(shape, dtype=dtype), dev)

    def _load_or_tune(tag, mod, work_dir, this_max_trials, space_gen=None):
        """Try cache -> tune -> return (lib | None, tuned: bool)."""
        tuning_record   = f"{work_dir}/database_tuning_record.json"
        workload_record = f"{work_dir}/database_workload.json"
        has_db = (
            os.path.exists(tuning_record)
            and os.path.exists(workload_record)
            and os.path.getsize(tuning_record) > 0
            and os.path.getsize(workload_record) > 0
        )

        def build_from_db():
            try:
                database = ms.database.JSONDatabase(work_dir=work_dir)
                sch = ms.tir_integration.compile_tir(database, mod, target)
                if sch is None:
                    return None
                with tvm.transform.PassContext(opt_level=3):
                    return tvm.build(sch.mod, target=target)
            except Exception as err:
                print(f"  [cache-invalid] {tag}: {type(err).__name__}: {err}")
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
                return None

        if has_db:
            print(f"  [cache] {tag}")
            lib = build_from_db()
            if lib is not None:
                return lib, True
            print(f"  [retune] {tag} cache invalid, retuning")

        if this_max_trials <= 0:
            print(f"  [no-cache] {tag} – no usable DB and max_trials=0")
            return None, False

        print(f"  [tune] {tag} up to {this_max_trials} trials")
        tune_kwargs: dict = dict(
            mod=mod,
            target=target,
            max_trials_global=this_max_trials,
            num_trials_per_iter=32,
            work_dir=work_dir,
        )
        if space_gen is not None:
            tune_kwargs["space"] = space_gen

        try:
            ms.tune_tir(**tune_kwargs)
            volume.commit()
            lib = build_from_db()
            if lib is not None:
                return lib, True
            print(f"  [tune-no-records] {tag} – tuner produced no valid schedules")
        except Exception as e:
            print(f"  [tune-error] {tag}: {e}")

        return None, False

    # ------------------------------------------------------------------
    # GEMM kernel  (work_dir kept under attn_v4_ so existing caches reused)
    # ------------------------------------------------------------------
    def make_gemm_mod(M, N, K):
        k_ax = te.reduce_axis((0, K), name="k")
        A = te.placeholder((M, K), name="A", dtype="float16")
        B = te.placeholder((K, N), name="B", dtype="float16")
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k_ax] * B[k_ax, j], axis=k_ax),
            name="C",
        )
        return tvm.IRModule({"main": te.create_prim_func([A, B, C])}), A, B, C

    gemm_max_trials = 0 if fused_only else max_trials

    def prepare_gemm(M, N, K, tag):
        # v4 prefix — backward compatible with existing tuned caches.
        work_dir = f"/tuning_results/attn_v4_{tag}_{M}_{N}_{K}_{arch}"
        mod, A_ph, B_ph, C_ph = make_gemm_mod(M, N, K)

        lib, tuned = _load_or_tune(tag, mod, work_dir, gemm_max_trials)
        if lib is not None:
            return lib, tuned

        # Simple tiled fallback
        print(f"  [fallback] {tag} simple tiled schedule")
        s = te.create_schedule(C_ph.op)
        io, ii = s[C_ph].split(s[C_ph].op.axis[0], factor=16)
        jo, ji = s[C_ph].split(s[C_ph].op.axis[1], factor=16)
        s[C_ph].bind(io, te.thread_axis("blockIdx.y"))
        s[C_ph].bind(jo, te.thread_axis("blockIdx.x"))
        s[C_ph].bind(ii, te.thread_axis("threadIdx.y"))
        s[C_ph].bind(ji, te.thread_axis("threadIdx.x"))
        with tvm.transform.PassContext(opt_level=3):
            return tvm.build(s, [A_ph, B_ph, C_ph], target=target), False

    # ------------------------------------------------------------------
    # Reshape kernel  (S, H*D) -> (H, S, D)
    # ------------------------------------------------------------------
    # Purely memory-bandwidth bound; hand-written schedule, no tuning.
    # One threadIdx.x per D element; blockIdx over (H, S).
    # ------------------------------------------------------------------
    def prepare_reshape(S, H, D, tag):
        flat  = te.placeholder((S, H * D), name="flat",  dtype="float16")
        heads = te.compute(
            (H, S, D),
            lambda h, s, d: flat[s, h * D + d],
            name="heads",
        )
        mod = tvm.IRModule({"main": te.create_prim_func([flat, heads])})

        sch    = tvm.tir.Schedule(mod)
        blk    = sch.get_block("heads")
        h, s, d = sch.get_loops(blk)
        factor = min(D, 128)
        do, di = sch.split(d, factors=[None, factor])
        sch.bind(h,  "blockIdx.z")
        sch.bind(s,  "blockIdx.y")
        sch.bind(do, "blockIdx.x")
        sch.bind(di, "threadIdx.x")
        print(f"  [reshape] {tag} ({S},{H},{D}) hand-written schedule")
        with tvm.transform.PassContext(opt_level=3):
            return tvm.build(sch.mod, target=target)

    # ------------------------------------------------------------------
    # QK GEMM kernel  — inputs in (H, S, D) layout
    # ------------------------------------------------------------------
    # Root cause of 0-trials in v5: Q_flat[i, h*D + k1] combined the
    # spatial loop h and the reduction axis k1 into one strided index.
    # MetaSchedule's MultiLevelTiling requires the reduce axis to map
    # cleanly to one input dimension (standard A[i,k]*B[k,j] pattern).
    # The strided access broke GEMM pattern recognition -> 0 candidates.
    #
    # Fix: accept Q[H,S,D] and K[H,S,D] directly.
    # QK[h,i,j] = inv_scale * sum_{d} Q[h,i,d] * K[h,j,d]
    # This is a clean batched GEMM; MultiLevelTiling fires on (i,j) x d.
    # ------------------------------------------------------------------
    def make_qk_mod(S, H, D, inv_sc):
        Q  = te.placeholder((H, S, D), name="Q",  dtype="float16")
        K  = te.placeholder((H, S, D), name="K",  dtype="float16")
        k1 = te.reduce_axis((0, D), "k1")
        inv_c = tvm.tir.const(inv_sc, "float16")
        QK = te.compute(
            (H, S, S),
            lambda h, i, j: te.sum(Q[h, i, k1] * K[h, j, k1] * inv_c, axis=k1),
            name="QK",
        )
        func = te.create_prim_func([Q, K, QK])
        return tvm.IRModule({"main": func}), Q, K, QK

    def qk_fallback_schedule(mod, S, H, D):
        sch = tvm.tir.Schedule(mod)
        blk = sch.get_block("QK")
        h, i, j, k1 = sch.get_loops(blk)
        sch.bind(h, "blockIdx.z")
        sch.bind(i, "blockIdx.y")
        factor = min(S, 128)
        jo, ji = sch.split(j, factors=[None, factor])
        sch.bind(jo, "blockIdx.x")
        sch.bind(ji, "threadIdx.x")
        with tvm.transform.PassContext(opt_level=3):
            return tvm.build(sch.mod, target=target)

    def prepare_qk(S, H, D, inv_sc):
        tag      = f"qk_gemm_v6_{S}_{H}_{D}_{arch}"
        work_dir = f"/tuning_results/attn_v6_{tag}"
        mod, _, _, _ = make_qk_mod(S, H, D, inv_sc)

        lib, tuned = _load_or_tune(tag, mod, work_dir, max_trials)
        if lib is not None:
            return lib, tuned

        print(f"  [fallback] {tag} tiled QK schedule")
        return qk_fallback_schedule(mod, S, H, D), False

    # ------------------------------------------------------------------
    # SoftmaxAV kernel  — V in (H, S, D) layout (no strided access)
    # ------------------------------------------------------------------
    # Previous: V_flat[k2, h*D + d] had same strided-access problem.
    # Fix: accept V[H,S,D] directly; AV is clean batched GEMM.
    # ------------------------------------------------------------------
    def make_softmax_av_mod(S, H, D):
        QK = te.placeholder((H, S, S), name="QK", dtype="float16")
        V  = te.placeholder((H, S, D), name="V",  dtype="float16")

        # --- row-max (numerically stable softmax) ---
        m_ax = te.reduce_axis((0, S), "m_ax")
        RowMax = te.compute(
            (H, S),
            lambda h, i: te.max(QK[h, i, m_ax], axis=m_ax),
            name="RowMax",
        )

        # --- exp(x - max) ---
        Exp = te.compute(
            (H, S, S),
            lambda h, i, j: tvm.te.exp(QK[h, i, j] - RowMax[h, i]),
            name="Exp",
        )

        # --- row-sum ---
        s_ax = te.reduce_axis((0, S), "s_ax")
        RowSum = te.compute(
            (H, S),
            lambda h, i: te.sum(Exp[h, i, s_ax], axis=s_ax),
            name="RowSum",
        )

        # --- normalised attention ---
        Attn = te.compute(
            (H, S, S),
            lambda h, i, j: Exp[h, i, j] / RowSum[h, i],
            name="Attn",
        )

        # --- AV: Attn @ V — clean (H, S, D), no strided V access ---
        k2 = te.reduce_axis((0, S), "k2")
        Out = te.compute(
            (H, S, D),
            lambda h, i, d: te.sum(Attn[h, i, k2] * V[h, k2, d], axis=k2),
            name="Out",
        )

        func = te.create_prim_func([QK, V, Out])
        return tvm.IRModule({"main": func}), QK, V, Out

    # Space generator: exclude CrossThreadReduction.
    # It fires on RowMax/RowSum before threadIdx.x is bound and crashes.
    def make_softmax_av_space_gen():
        from tvm.meta_schedule.space_generator import PostOrderApply
        from tvm.meta_schedule import schedule_rule as sr
        all_rules = sr.ScheduleRule.create("cuda")
        filtered  = [r for r in all_rules
                     if "CrossThreadReduction" not in type(r).__name__]
        return PostOrderApply(sch_rules=filtered)

    # Fallback: compute_at pulls RowMax/RowSum inside Out tile -> no race, no NaN.
    def softmax_av_fallback_schedule(mod):
        sch = tvm.tir.Schedule(mod)

        out_block    = sch.get_block("Out")
        attn_block   = sch.get_block("Attn")
        rowsum_block = sch.get_block("RowSum")
        exp_block    = sch.get_block("Exp")
        rowmax_block = sch.get_block("RowMax")

        # Bind (h, i) of Out to grid; d and k2 remain serial.
        h_loop, i_loop, d_loop, k2_loop = sch.get_loops(out_block)
        sch.bind(h_loop, "blockIdx.z")
        sch.bind(i_loop, "blockIdx.x")

        # Pull producers inside i_loop: sequential within the block,
        # correct ordering guaranteed without any explicit barrier.
        sch.compute_at(attn_block,   i_loop)
        sch.compute_at(rowsum_block, i_loop)
        sch.compute_at(exp_block,    i_loop)
        sch.compute_at(rowmax_block, i_loop)

        with tvm.transform.PassContext(opt_level=3):
            return tvm.build(sch.mod, target=target)

    def prepare_softmax_av(S, H, D):
        tag      = f"softmax_av_v6_{S}_{H}_{D}_{arch}"
        work_dir = f"/tuning_results/attn_v6_{tag}"
        mod, _, _, _ = make_softmax_av_mod(S, H, D)

        lib, tuned = _load_or_tune(
            tag, mod, work_dir, max_trials,
            space_gen=make_softmax_av_space_gen(),
        )
        if lib is not None:
            return lib, tuned

        print(f"  [fallback] {tag} compute_at fallback (NaN-safe)")
        return softmax_av_fallback_schedule(mod), False

    # ------------------------------------------------------------------
    # Build / load all kernels
    # ------------------------------------------------------------------
    q_lib  = k_lib  = v_lib  = qk_lib  = av_lib  = None
    reshape_q_lib = reshape_k_lib = reshape_v_lib = None
    fused_qk_lib  = fused_sav_lib = None

    q_tuned = k_tuned = v_tuned = qk_tuned = av_tuned = False
    fused_qk_tuned = fused_sav_tuned = False

    if not pytorch_only:
        # Existing component GEMM kernels (v4 cache prefix preserved)
        q_lib,  q_tuned  = prepare_gemm(seq_len, hidden_dim, hidden_dim, "Q_proj")
        k_lib,  k_tuned  = prepare_gemm(seq_len, hidden_dim, hidden_dim, "K_proj")
        v_lib,  v_tuned  = prepare_gemm(seq_len, hidden_dim, hidden_dim, "V_proj")
        qk_lib, qk_tuned = prepare_gemm(seq_len, seq_len,   head_dim,   "QK_dot")
        av_lib, av_tuned = prepare_gemm(seq_len, head_dim,  seq_len,    "AV_sum")

        # Reshape: (S, H*D) -> (H, S, D) — hand-written, not tuned
        reshape_q_lib = prepare_reshape(seq_len, num_heads, head_dim, "reshape_Q")
        reshape_k_lib = prepare_reshape(seq_len, num_heads, head_dim, "reshape_K")
        reshape_v_lib = prepare_reshape(seq_len, num_heads, head_dim, "reshape_V")

        # Fused QK GEMM: (H,S,D) x (H,S,D) -> (H,S,S)
        fused_qk_lib,  fused_qk_tuned  = prepare_qk(seq_len, num_heads, head_dim, inv_scale)
        # Fused SoftmaxAV: (H,S,S) x (H,S,D) -> (H,S,D)
        fused_sav_lib, fused_sav_tuned = prepare_softmax_av(seq_len, num_heads, head_dim)

    results = {
        "model":      model,
        "seq_len":    seq_len,
        "hidden_dim": hidden_dim,
        "num_heads":  num_heads,
        "gpu":        gpu_name,
        "arch":       arch,
        "pytorch_only":    pytorch_only,
        "tuned":           all([q_tuned, k_tuned, v_tuned, qk_tuned, av_tuned]) if not pytorch_only else None,
        "fused_qk_tuned":  fused_qk_tuned  if not pytorch_only else None,
        "fused_sav_tuned": fused_sav_tuned if not pytorch_only else None,
    }

    # ------------------------------------------------------------------
    # PyTorch baselines
    # ------------------------------------------------------------------
    print("\n[PyTorch baselines]")

    def pt_qkv():
        return X @ Wq, X @ Wk, X @ Wv

    results["pt_qkv_ms"] = _timeit(pt_qkv, iters=iters)

    Q_pt, K_pt, V_pt = pt_qkv()
    Q_mh = Q_pt.view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()
    K_mh = K_pt.view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()
    V_mh = V_pt.view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()

    def pt_dot_batched():
        return torch.bmm(Q_mh, K_mh.transpose(-2, -1))

    def pt_dot_loop():
        return [Q_mh[h] @ K_mh[h].transpose(-2, -1) for h in range(num_heads)]

    results["pt_dot_batched_ms"] = _timeit(pt_dot_batched, iters=iters)
    results["pt_dot_loop_ms"]    = _timeit(pt_dot_loop,    iters=iters)

    scores_batched = pt_dot_batched()

    def pt_softmax_batched():
        return F.softmax(scores_batched / scale, dim=-1)

    results["pt_softmax_ms"] = _timeit(pt_softmax_batched, iters=iters)
    attn_batched = pt_softmax_batched()

    def pt_av_batched():
        return torch.bmm(attn_batched, V_mh)

    def pt_av_loop():
        return [attn_batched[h] @ V_mh[h] for h in range(num_heads)]

    results["pt_av_batched_ms"] = _timeit(pt_av_batched, iters=iters)
    results["pt_av_loop_ms"]    = _timeit(pt_av_loop,    iters=iters)

    def pt_full_unfused_batched():
        Q_ = (X @ Wq).view(seq_len, num_heads, head_dim).transpose(0, 1)
        K_ = (X @ Wk).view(seq_len, num_heads, head_dim).transpose(0, 1)
        V_ = (X @ Wv).view(seq_len, num_heads, head_dim).transpose(0, 1)
        return torch.bmm(
            F.softmax(torch.bmm(Q_, K_.transpose(-2, -1)) / scale, dim=-1), V_
        )

    def pt_sdpa():
        Q_ = (X @ Wq).view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        K_ = (X @ Wk).view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        V_ = (X @ Wv).view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        return F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)

    results["pt_full_unfused_batched_ms"] = _timeit(pt_full_unfused_batched, iters=iters)
    results["pt_sdpa_ms"]                 = _timeit(pt_sdpa,                 iters=iters)

    results["pt_pipeline_loop_no_softmax_ms"] = (
        results["pt_qkv_ms"] + results["pt_dot_loop_ms"] + results["pt_av_loop_ms"]
    )
    results["pt_pipeline_loop_with_softmax_ms"] = (
        results["pt_pipeline_loop_no_softmax_ms"] + results["pt_softmax_ms"]
    )

    if pytorch_only:
        print("[skip] TVM kernels (pytorch_only=True)")
        return results

    # ------------------------------------------------------------------
    # TVM component pipeline
    # ------------------------------------------------------------------
    print("[TVM component pipeline]")

    X_tvm  = to_tvm(X)
    Wq_tvm = to_tvm(Wq)
    Wk_tvm = to_tvm(Wk)
    Wv_tvm = to_tvm(Wv)
    Q_tvm  = zeros_tvm((seq_len, hidden_dim))
    K_tvm  = zeros_tvm((seq_len, hidden_dim))
    V_tvm  = zeros_tvm((seq_len, hidden_dim))

    def tvm_qkv():
        q_lib(X_tvm, Wq_tvm, Q_tvm)
        k_lib(X_tvm, Wk_tvm, K_tvm)
        v_lib(X_tvm, Wv_tvm, V_tvm)
        dev.sync()

    results["tvm_qkv_ms"] = _timeit(tvm_qkv, iters=iters)

    Q_heads_tvm  = [to_tvm(Q_mh[h]) for h in range(num_heads)]
    Kt_heads_tvm = [to_tvm(K_mh[h].transpose(-2, -1).contiguous()) for h in range(num_heads)]
    QK_outs_tvm  = [zeros_tvm((seq_len, seq_len)) for _ in range(num_heads)]
    AV_in_tvm    = [to_tvm(attn_batched[h]) for h in range(num_heads)]
    V_heads_comp = [to_tvm(V_mh[h]) for h in range(num_heads)]
    AV_outs_tvm  = [zeros_tvm((seq_len, head_dim)) for _ in range(num_heads)]

    def tvm_dot_loop():
        for h in range(num_heads):
            qk_lib(Q_heads_tvm[h], Kt_heads_tvm[h], QK_outs_tvm[h])
        dev.sync()

    def tvm_av_loop():
        for h in range(num_heads):
            av_lib(AV_in_tvm[h], V_heads_comp[h], AV_outs_tvm[h])
        dev.sync()

    results["tvm_dot_loop_ms"] = _timeit(tvm_dot_loop, iters=iters)
    results["tvm_av_loop_ms"]  = _timeit(tvm_av_loop,  iters=iters)

    results["tvm_pipeline_loop_no_softmax_ms"] = (
        results["tvm_qkv_ms"] + results["tvm_dot_loop_ms"] + results["tvm_av_loop_ms"]
    )
    results["tvm_pipeline_loop_with_softmax_ms"] = (
        results["tvm_pipeline_loop_no_softmax_ms"] + results["pt_softmax_ms"]
    )

    results["qkv_speedup_pt_over_tvm"]  = results["pt_qkv_ms"]  / results["tvm_qkv_ms"]
    results["dot_loop_speedup_pt_over_tvm"] = results["pt_dot_loop_ms"] / results["tvm_dot_loop_ms"]
    results["av_loop_speedup_pt_over_tvm"]  = results["pt_av_loop_ms"]  / results["tvm_av_loop_ms"]
    results["pipeline_no_softmax_speedup_pt_over_tvm"] = (
        results["pt_pipeline_loop_no_softmax_ms"] / results["tvm_pipeline_loop_no_softmax_ms"]
    )

    print(
        f"  QKV:     PT {results['pt_qkv_ms']:.4f} ms  TVM {results['tvm_qkv_ms']:.4f} ms "
        f"({results['qkv_speedup_pt_over_tvm']:.2f}x PT/TVM)"
    )
    print(
        f"  QK loop: PT {results['pt_dot_loop_ms']:.4f} ms  TVM {results['tvm_dot_loop_ms']:.4f} ms "
        f"({results['dot_loop_speedup_pt_over_tvm']:.2f}x PT/TVM)"
    )
    print(
        f"  AV loop: PT {results['pt_av_loop_ms']:.4f} ms  TVM {results['tvm_av_loop_ms']:.4f} ms "
        f"({results['av_loop_speedup_pt_over_tvm']:.2f}x PT/TVM)"
    )

    # ------------------------------------------------------------------
    # TVM fused SDPA  (reshape -> QK GEMM -> SoftmaxAV)
    # ------------------------------------------------------------------
    if (fused_qk_lib is not None
            and fused_sav_lib is not None
            and reshape_q_lib is not None):

        print("\n[TVM fused SDPA  (reshape -> QK GEMM -> SoftmaxAV)]")

        # Flat projection outputs  (S, H*D)
        Q_flat_tvm = zeros_tvm((seq_len, hidden_dim))
        K_flat_tvm = zeros_tvm((seq_len, hidden_dim))
        V_flat_tvm = zeros_tvm((seq_len, hidden_dim))

        # Reshaped head tensors  (H, S, D)
        Q_heads_fused = zeros_tvm((num_heads, seq_len, head_dim))
        K_heads_fused = zeros_tvm((num_heads, seq_len, head_dim))
        V_heads_fused = zeros_tvm((num_heads, seq_len, head_dim))

        # Intermediate and output buffers
        QK_out_tvm    = zeros_tvm((num_heads, seq_len, seq_len))
        fused_out_tvm = zeros_tvm((num_heads, seq_len, head_dim))

        def tvm_fused_sdpa():
            # Step 1: QKV projections  -> (S, H*D)
            q_lib(X_tvm, Wq_tvm, Q_flat_tvm)
            k_lib(X_tvm, Wk_tvm, K_flat_tvm)
            v_lib(X_tvm, Wv_tvm, V_flat_tvm)

            # Step 2: Reshape  (S, H*D) -> (H, S, D)
            reshape_q_lib(Q_flat_tvm, Q_heads_fused)
            reshape_k_lib(K_flat_tvm, K_heads_fused)
            reshape_v_lib(V_flat_tvm, V_heads_fused)

            # Step 3: Scaled QK GEMM  -> (H, S, S)
            fused_qk_lib(Q_heads_fused, K_heads_fused, QK_out_tvm)

            # Step 4: Softmax + AV    -> (H, S, D)
            fused_sav_lib(QK_out_tvm, V_heads_fused, fused_out_tvm)
            dev.sync()

        # --- Correctness check ---
        tvm_fused_sdpa()
        fused_out_np = fused_out_tvm.numpy().astype("float32")            # (H, S, D)
        pt_ref_np    = pt_sdpa().squeeze(0).detach().cpu().float().numpy() # (H, S, D)
        ok = np.allclose(fused_out_np, pt_ref_np, rtol=1e-2, atol=1e-2)
        results["fused_correctness_ok"] = bool(ok)
        if ok:
            print("  [correctness] TVM fused SDPA matches PyTorch SDPA ✓")
        else:
            max_err  = np.abs(fused_out_np - pt_ref_np).max()
            mean_err = np.abs(fused_out_np - pt_ref_np).mean()
            print(f"  [correctness] MISMATCH (max_err={max_err:.4f}, "
                  f"mean_err={mean_err:.6f}) ✗")

        # --- Benchmark ---
        results["tvm_fused_sdpa_ms"] = _timeit(tvm_fused_sdpa, iters=iters)

        results["fused_speedup_pt_sdpa_over_tvm_fused"] = (
            results["tvm_fused_sdpa_ms"] / results["pt_sdpa_ms"]
        )
        results["fused_speedup_tvm_pipeline_over_fused"] = (
            results["tvm_pipeline_loop_with_softmax_ms"] / results["tvm_fused_sdpa_ms"]
        )

        qk_s  = "tuned"   if fused_qk_tuned  else "fallback"
        sav_s = "tuned"   if fused_sav_tuned else "fallback"
        print(
            f"  PT SDPA:         {results['pt_sdpa_ms']:.4f} ms  "
            f"(NOTE: may use FlashAttention-2 internally)"
        )
        print(
            f"  TVM fused SDPA:  {results['tvm_fused_sdpa_ms']:.4f} ms  "
            f"(QK:{qk_s} SAV:{sav_s}) "
            f"| TVM/PT = {results['fused_speedup_pt_sdpa_over_tvm_fused']:.2f}x"
        )
        print(
            f"  TVM pipeline:    {results['tvm_pipeline_loop_with_softmax_ms']:.4f} ms"
            f" | pipeline/fused = {results['fused_speedup_tvm_pipeline_over_fused']:.2f}x"
        )
    else:
        print("  [skip] Fused attn kernel(s) unavailable")
        results.update({
            "tvm_fused_sdpa_ms":                     None,
            "fused_correctness_ok":                  None,
            "fused_speedup_pt_sdpa_over_tvm_fused":  None,
            "fused_speedup_tvm_pipeline_over_fused": None,
        })

    return results


@app.local_entrypoint()
def main(
    max_trials: int = 800,
    iters: int = 200,
    pytorch_only: bool = False,
    fused_only: bool = False,
):
    import json

    print("=" * 80)
    print("TVM vs PyTorch Attention Benchmark v6 (Correct Layout)")
    print("=" * 80)
    print(f"Shapes: {ATTENTION_SHAPES}")
    print(f"Trials: {max_trials} per kernel")
    print(f"Iters : {iters}")
    print(f"PyTorch only: {pytorch_only}")
    print(f"Fused only:   {fused_only}  "
          f"(GEMM kernels from v4 cache; only fused kernels tuned)")
    print("=" * 80)

    all_results = list(
        tune_and_benchmark_attention_v6.starmap(
            [
                (s, h, n, m, max_trials, iters, pytorch_only, fused_only)
                for s, h, n, m in ATTENTION_SHAPES
            ]
        )
    )

    has_tvm = all_results and "tvm_qkv_ms" in all_results[0]
    if has_tvm:
        print("\n" + "=" * 120)
        print("MATCHED (PER-HEAD LOOP) COMPARISON")
        print("=" * 120)
        hdr = (
            f"{'Model':<7} {'Seq':>5} "
            f"{'PT QKV':>8} {'TVM QKV':>8} {'PT/TVM':>8} "
            f"{'PT QK(loop)':>12} {'TVM QK(loop)':>13} {'PT/TVM':>8} "
            f"{'PT AV(loop)':>12} {'TVM AV(loop)':>13} {'PT/TVM':>8} "
            f"{'PT Pipe(no sfmx)':>16} {'TVM Pipe(no sfmx)':>17} {'PT/TVM':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in all_results:
            print(
                f"{r['model']:<7} {r['seq_len']:>5} "
                f"{r['pt_qkv_ms']:>8.3f} {r['tvm_qkv_ms']:>8.3f} "
                f"{r['qkv_speedup_pt_over_tvm']:>7.2f}x "
                f"{r['pt_dot_loop_ms']:>12.3f} {r['tvm_dot_loop_ms']:>13.3f} "
                f"{r['dot_loop_speedup_pt_over_tvm']:>7.2f}x "
                f"{r['pt_av_loop_ms']:>12.3f} {r['tvm_av_loop_ms']:>13.3f} "
                f"{r['av_loop_speedup_pt_over_tvm']:>7.2f}x "
                f"{r['pt_pipeline_loop_no_softmax_ms']:>16.3f} "
                f"{r['tvm_pipeline_loop_no_softmax_ms']:>17.3f} "
                f"{r['pipeline_no_softmax_speedup_pt_over_tvm']:>7.2f}x"
            )

    print("\n" + "=" * 105)
    print("PYTORCH BEST-PATH REFERENCE")
    print("=" * 105)
    hdr2 = (
        f"{'Model':<7} {'Seq':>5} {'PT Dot(bmm)':>12} {'PT Dot(loop)':>13} "
        f"{'PT AV(bmm)':>11} {'PT AV(loop)':>12} {'PT Full Unfused':>15} {'PT SDPA':>10}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for r in all_results:
        print(
            f"{r['model']:<7} {r['seq_len']:>5} "
            f"{r['pt_dot_batched_ms']:>12.3f} {r['pt_dot_loop_ms']:>13.3f} "
            f"{r['pt_av_batched_ms']:>11.3f} {r['pt_av_loop_ms']:>12.3f} "
            f"{r['pt_full_unfused_batched_ms']:>15.3f} {r['pt_sdpa_ms']:>10.3f}"
        )

    has_fused = has_tvm and all_results and all_results[0].get("tvm_fused_sdpa_ms") is not None
    if has_fused:
        print("\n" + "=" * 110)
        print("FUSED SDPA COMPARISON  (apples-to-apples: same logical computation)")
        print("  PT SDPA may use FlashAttention-2; TVM fused materialises full attn matrix.")
        print("=" * 110)
        hdr3 = (
            f"{'Model':<7} {'Seq':>5} "
            f"{'PT SDPA':>10} {'PT Unfused':>11} {'TVM Pipeline':>14} "
            f"{'TVM Fused':>11} {'Correct':>8} "
            f"{'TVM/PT':>8} {'Pipe/Fused':>11}"
        )
        print(hdr3)
        print("-" * len(hdr3))
        for r in all_results:
            rt = r.get("fused_speedup_pt_sdpa_over_tvm_fused")
            rp = r.get("fused_speedup_tvm_pipeline_over_fused")
            ok = r.get("fused_correctness_ok")
            tf = r.get("tvm_fused_sdpa_ms")
            pm = r.get("tvm_pipeline_loop_with_softmax_ms")
            rt_s = f"{rt:>7.2f}x"  if rt is not None else "     N/A"
            rp_s = f"{rp:>10.2f}x" if rp is not None else "       N/A"
            tf_s = f"{tf:>11.3f}"  if tf is not None else "        N/A"
            pm_s = f"{pm:>14.3f}"  if pm is not None else "           N/A"
            ok_s = ("✓" if ok else "✗") if ok is not None else "?"
            print(
                f"{r['model']:<7} {r['seq_len']:>5} "
                f"{r['pt_sdpa_ms']:>10.3f} {r['pt_full_unfused_batched_ms']:>11.3f} "
                f"{pm_s} {tf_s} {ok_s:>8} "
                f"{rt_s} {rp_s}"
            )
        print()
        print("  TVM/PT  >1 means TVM fused is SLOWER than PT SDPA.")
        print("  Pipe/Fused >1 means TVM fused is FASTER than the TVM component pipeline.")

    out_path = Path(__file__).parent.parent / "results" / "attention_results_v6.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pytorch_only:
        pt_keys = {
            "gpu", "arch", "pytorch_only",
            "pt_qkv_ms", "pt_dot_batched_ms", "pt_dot_loop_ms",
            "pt_softmax_ms", "pt_av_batched_ms", "pt_av_loop_ms",
            "pt_full_unfused_batched_ms", "pt_sdpa_ms",
            "pt_pipeline_loop_no_softmax_ms", "pt_pipeline_loop_with_softmax_ms",
        }
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = []

        index = {
            (r.get("model"), r.get("seq_len"),
             r.get("hidden_dim"), r.get("num_heads")): r
            for r in existing
        }
        for r in all_results:
            key = (r.get("model"), r.get("seq_len"),
                   r.get("hidden_dim"), r.get("num_heads"))
            if key in index:
                for k in pt_keys:
                    if k in r:
                        index[key][k] = r[k]
            else:
                index[key] = r

        merged = sorted(
            index.values(),
            key=lambda x: (str(x.get("model", "")), int(x.get("seq_len", 0)),
                           int(x.get("hidden_dim", 0)), int(x.get("num_heads", 0))),
        )
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nSaved: {out_path} (PyTorch fields merged; TVM fields preserved)")
    else:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out_path}")