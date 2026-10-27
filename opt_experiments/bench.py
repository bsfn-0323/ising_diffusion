"""
A/B benchmark of optimization variants for GNNUnet (unetGnn_cfg.py) — source
untouched.  Every variant reuses the SAME baseline weights (shared submodules),
so we verify each produces baseline-identical output before trusting its timing.

Variants (optimizations #2,#3,#4,#5 — #1 dense-adjacency intentionally excluded):
  baseline        original GNNUnet(forward takes PyG Batch)
  plain           decoupled forward (plain tensors) — removes PyG dispatch + graph breaks
  aot_eager       torch.compile(plain, backend=aot_eager)  [#2, no Triton]
  cudagraphs-dyn  torch.compile(plain, backend=cudagraphs)  [#2/#5, no Triton]
  cudagraph-man   manual torch.cuda.CUDAGraph capture       [#5, no Triton]
  checkpoint      gradient-checkpoint each iter             [#3, memory]
  jit-fused       fused pointwise via torch.jit.script      [#4]
  jit+cudagraph   combine #4 + #5

Reports fwd ms, fwd+bwd ms, peak MB (fwd+bwd), and max|Δ| vs baseline output.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import torch
from profile_unetgnn_cfg import (build_model, build_rrg, make_batch, x_input_of,
                                 time_fn, DEVICE, N_TRAIN, B_TRAIN, TIMESTEPS)
from opt_experiments.plain_model import GNNUnetPlain, unpack_batch


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def peak_fwdbwd(run_fwdbwd):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    run_fwdbwd()
    sync()
    return torch.cuda.max_memory_allocated() / 1024**2


# ── manual CUDA graph capture ─────────────────────────────────────────────────
def capture_fwd(plain, args):
    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in args.items()}
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            plain(**static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        out = plain(**static)
    return (lambda: g.replay()), static, out


def capture_fwdbwd(plain, args):
    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in args.items()}
    for p in plain.parameters():
        p.grad = None
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = plain(**static)
            out.float().pow(2).mean().backward()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = plain(**static)
        out.float().pow(2).mean().backward()
    return (lambda: g.replay()), static


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    torch.manual_seed(0)
    base = build_model()
    ei = build_rrg(N_TRAIN, 3)
    batch = make_batch(B_TRAIN, ei, N_TRAIN)
    t = torch.randint(0, TIMESTEPS - 1, (B_TRAIN,), device=DEVICE).float()
    args = dict(unpack_batch(batch), t=t)
    print(f"Config: N={N_TRAIN} B={B_TRAIN}  nodes={batch.num_nodes} params="
          f"{sum(p.numel() for p in base.parameters()):,}\n")

    # reference output
    with torch.no_grad():
        ref = base(x_input_of(batch), batch, t).float()

    results = []  # (name, fwd_ms, fwdbwd_ms, peakMB, maxdiff, note)

    # ---- baseline ----
    def b_fwd():
        with torch.no_grad():
            base(x_input_of(batch), batch, t)
    def b_fb():
        out = base(x_input_of(batch), batch, t)
        out.float().pow(2).mean().backward(); base.zero_grad(set_to_none=True)
    results.append(("baseline (PyG Batch)", time_fn(b_fwd), time_fn(b_fb),
                    peak_fwdbwd(b_fb), 0.0, ""))

    # ---- plain (decoupled) ----
    plain = GNNUnetPlain(base).to(DEVICE)
    with torch.no_grad():
        out = plain(**args).float()
    d = (out - ref).abs().max().item()
    def p_fwd():
        with torch.no_grad(): plain(**args)
    def p_fb():
        out = plain(**args); out.float().pow(2).mean().backward()
        plain.zero_grad(set_to_none=True)
    results.append(("plain (no PyG dispatch)", time_fn(p_fwd), time_fn(p_fb),
                    peak_fwdbwd(p_fb), d, ""))

    # ---- aot_eager compile ----
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        cm = torch.compile(plain, backend="aot_eager", dynamic=False)
        with torch.no_grad():
            o = cm(**args).float()
        d = (o - ref).abs().max().item()
        def c_fwd():
            with torch.no_grad(): cm(**args)
        def c_fb():
            out = cm(**args); out.float().pow(2).mean().backward()
            plain.zero_grad(set_to_none=True)
        results.append(("compile aot_eager", time_fn(c_fwd), time_fn(c_fb),
                        peak_fwdbwd(c_fb), d, ""))
    except Exception as e:
        results.append(("compile aot_eager", 0, 0, 0, 0, f"FAIL {type(e).__name__}"))

    # ---- cudagraphs dynamo backend ----
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        cg = torch.compile(plain, backend="cudagraphs", dynamic=False)
        with torch.no_grad():
            o = cg(**args).float()
        d = (o - ref).abs().max().item()
        def cg_fwd():
            with torch.no_grad(): cg(**args)
        results.append(("compile cudagraphs", time_fn(cg_fwd), float("nan"),
                        float("nan"), d, "fwd-only"))
    except Exception as e:
        results.append(("compile cudagraphs", 0, 0, 0, 0, f"FAIL {type(e).__name__}"))

    # ---- manual CUDA graph (fwd) ----
    try:
        run, static, gout = capture_fwd(GNNUnetPlain(base).to(DEVICE), args)
        # copy real inputs -> static, replay, check parity
        with torch.no_grad():
            for k, v in args.items():
                if torch.is_tensor(v):
                    static[k].copy_(v)
        run(); sync()
        d = (gout.float() - ref).abs().max().item()
        fwd_ms = time_fn(run, warmup=5, reps=60)
        # fwd+bwd graph
        try:
            runfb, _ = capture_fwdbwd(GNNUnetPlain(base).to(DEVICE), args)
            fb_ms = time_fn(runfb, warmup=5, reps=60)
        except Exception as e:
            fb_ms = float("nan")
        results.append(("cudagraph manual", fwd_ms, fb_ms, float("nan"), d, ""))
    except Exception as e:
        results.append(("cudagraph manual", 0, 0, 0, 0, f"FAIL {type(e).__name__}: {e}"))

    # ---- gradient checkpoint ----
    plain_ck = GNNUnetPlain(base, use_checkpoint=True).to(DEVICE)
    with torch.no_grad():
        o = plain_ck(**args).float()
    d = (o - ref).abs().max().item()
    def ck_fwd():
        with torch.no_grad(): plain_ck(**args)
    def ck_fb():
        out = plain_ck(**args); out.float().pow(2).mean().backward()
        plain_ck.zero_grad(set_to_none=True)
    results.append(("checkpoint (mem)", time_fn(ck_fwd), time_fn(ck_fb),
                    peak_fwdbwd(ck_fb), d, ""))

    # ---- jit fused pointwise ----
    plain_jit = GNNUnetPlain(base, use_jit_fused=True).to(DEVICE)
    with torch.no_grad():
        for _ in range(3):
            plain_jit(**args)          # warm the jit fuser
        o = plain_jit(**args).float()
    d = (o - ref).abs().max().item()
    def j_fwd():
        with torch.no_grad(): plain_jit(**args)
    def j_fb():
        out = plain_jit(**args); out.float().pow(2).mean().backward()
        plain_jit.zero_grad(set_to_none=True)
    results.append(("jit-fused pointwise", time_fn(j_fwd), time_fn(j_fb),
                    peak_fwdbwd(j_fb), d, ""))

    # ---- jit fused + manual cuda graph ----
    try:
        pj = GNNUnetPlain(base, use_jit_fused=True).to(DEVICE)
        with torch.no_grad():
            for _ in range(3): pj(**args)
        run, static, gout = capture_fwd(pj, args)
        with torch.no_grad():
            for k, v in args.items():
                if torch.is_tensor(v): static[k].copy_(v)
        run(); sync()
        d = (gout.float() - ref).abs().max().item()
        results.append(("jit-fused + cudagraph", time_fn(run, 5, 60), float("nan"),
                        float("nan"), d, "fwd-only"))
    except Exception as e:
        results.append(("jit-fused + cudagraph", 0, 0, 0, 0, f"FAIL {type(e).__name__}"))

    # ── report ────────────────────────────────────────────────────────────────
    base_fwd = results[0][1]; base_fb = results[0][2]
    print(f"{'variant':<26}{'fwd ms':>9}{'x':>6}{'fwd+bwd':>10}{'x':>6}"
          f"{'peakMB':>9}{'max|Δ|':>10}  note")
    print("-" * 92)
    for name, fw, fb, mb, dd, note in results:
        sp_f = f"{base_fwd/fw:.2f}" if fw else "—"
        sp_b = f"{base_fb/fb:.2f}" if (fb and fb == fb) else "—"
        fbs = f"{fb:.2f}" if fb == fb else "nan"
        mbs = f"{mb:.0f}" if mb == mb else "—"
        print(f"{name:<26}{fw:>9.2f}{sp_f:>6}{fbs:>10}{sp_b:>6}"
              f"{mbs:>9}{dd:>10.2e}  {note}")
    print("-" * 92)
    print("x = speedup vs baseline. max|Δ| = max abs diff of forward output vs baseline "
          "(want ~1e-5 fp noise).")


if __name__ == "__main__":
    main()
