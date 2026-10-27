"""
From-scratch profiler for models/unetGnn_cfg.py  (GNNUnet, recurrent GNN for RFIM diffusion).

Goals
-----
Find every optimization opportunity by measuring, on the *actual* GPU:
  (a) component-level ms breakdown of forward and forward+backward,
  (b) per-iteration breakdown inside the num_iters recurrent loop,
  (c) torch.profiler top ops by cuda_time_total (record_shapes, profile_memory),
  (d) peak GPU memory,
  (e) scaling across N to expose the sparse (PyG scatter) vs dense (matmul) crossover,
  (f) torch.compile graph-break count + speedup (data point for optimization proposal).

Config mirrors train_cfg.py: base_ch=64, num_iters=10, time_emb_dim=32,
batch_size=1024, L=12 -> N=144.  Model is NOT modified.

Graph: degree-3 undirected (ring + random perfect matching), symmetric edge_index,
degree-normalized edge weights * per-graph inv_temp -- exactly like MultiFieldIsingDataset.
"""
import os
import sys
import time
import argparse
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.profiler import profile, record_function, ProfilerActivity

from models.unetGnn_cfg import GNNUnet, rms_norm_fixed_N

# ── config (train_cfg.py defaults) ─────────────────────────────────────────────
BASE_CH    = 64
NUM_ITERS  = 10
TIME_EMB   = 32
L          = 12
N_TRAIN    = L * L          # 144
B_TRAIN    = 1024
TIMESTEPS  = 1000
BETA_SCALE = 100.0          # GNNUnet default
DEGREE     = 3              # RRG degree from Jmat.bin
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP     = 15
REPS       = 60

# For the N-scaling sweep keep total work (B*N) ~constant so we isolate the
# effect of per-graph size N on the sparse/dense ops (and avoid OOM at N=10k).
TOTAL_NODES = N_TRAIN * B_TRAIN     # 147456


# ── helpers ────────────────────────────────────────────────────────────────────
def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def build_rrg(N, degree=3, seed=0):
    """Degree-`degree` undirected graph as symmetric edge_index [2, E].

    Construction: a ring (degree 2) + one random perfect matching per extra
    degree (needs N even).  Cheap, vectorized, connected, ~degree-regular.
    Occasional coincidental parallel edges are fine for profiling shapes.
    """
    assert N % 2 == 0, "need even N for perfect matching"
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(N)

    src = [idx, idx]
    dst = [(idx + 1) % N, (idx - 1) % N]          # ring => degree 2

    for _ in range(degree - 2):                    # each matching adds degree 1
        perm = torch.randperm(N, generator=g)
        a, b = perm[0::2], perm[1::2]
        src += [a, b]
        dst += [b, a]

    src = torch.cat(src)
    dst = torch.cat(dst)
    return torch.stack([src, dst], dim=0).contiguous()


def make_batch(bs, edge_index_single, N):
    """Build a realistic Batch mirroring MultiFieldIsingDataset.get()."""
    src, dst = edge_index_single
    deg = torch.zeros(N).scatter_add_(0, dst, torch.ones(src.shape[0]))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_norm = (deg_inv_sqrt[src] * deg_inv_sqrt[dst]).contiguous()   # [E]

    graphs = []
    for _ in range(bs):
        x          = torch.randint(0, 2, (N, 1)).float() * 2 - 1
        field      = torch.randn(N, 1) * 0.5
        mk         = torch.randn(N, 1) * 0.5
        beta       = torch.rand(1) * 0.5 + 0.2          # inv_temp in [0.2, 0.7]
        field_mean = field.mean().view(1)
        field_std  = field.std().view(1).clamp(min=1e-6)
        edge_weight = (beta * deg_norm).contiguous()
        graphs.append(Data(
            x=x, mk=mk, field=field,
            field_mean=field_mean, field_std=field_std,
            edge_index=edge_index_single.clone(),
            edge_weight=edge_weight,
            inv_temp=beta,
            instance_idx=torch.tensor(0),
        ))
    return Batch.from_data_list(graphs).to(DEVICE)


def build_model():
    m = GNNUnet(base_ch=BASE_CH, time_emb_dim=TIME_EMB,
                discrete=True, num_iters=NUM_ITERS, beta_scale=BETA_SCALE).to(DEVICE)
    return m


def x_input_of(batch):
    return torch.cat([batch.x, batch.field], dim=1)     # [N_total, 2]


# ── accurate aggregate timing via CUDA events (no per-op serialization) ────────
def time_fn(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    sync()
    if DEVICE.type == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / reps
    else:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) * 1e3 / reps


# ── sync-based section timer (captures launch overhead; serializes GPU) ────────
class SectionTimer:
    def __init__(self):
        self.totals, self.counts = {}, {}
        self.order = []

    @contextlib.contextmanager
    def __call__(self, name):
        sync()
        t0 = time.perf_counter()
        yield
        sync()
        dt = (time.perf_counter() - t0) * 1e3
        if name not in self.totals:
            self.totals[name] = 0.0
            self.counts[name] = 0
            self.order.append(name)
        self.totals[name] += dt
        self.counts[name] += 1

    def report(self, reps, title):
        grand = sum(self.totals.values())
        print(f"\n{'─'*66}")
        print(f"{title}")
        print(f"{'─'*66}")
        print(f"{'Component':<42}{'ms':>9}{'%':>8}")
        print(f"{'─'*66}")
        for k in self.order:
            avg = self.totals[k] / reps
            pct = 100.0 * self.totals[k] / grand
            print(f"{k:<42}{avg:9.4f}{pct:7.1f}%")
        print(f"{'─'*66}")
        print(f"{'TOTAL (wall, serialized)':<42}{grand/reps:9.4f}{100.0:7.1f}%")
        print(f"{'─'*66}")


# ════════════════════════════════════════════════════════════════════════════
#  1. COMPONENT-LEVEL BREAKDOWN  (mirrors GNNUnet.forward exactly)
# ════════════════════════════════════════════════════════════════════════════
def component_breakdown(model, batch, t, backward=False):
    tm = SectionTimer()
    inv_temp   = batch.inv_temp
    num_graphs = batch.num_graphs
    batch_vec  = batch.batch
    N_per      = batch.num_nodes // num_graphs
    edge_index = batch.edge_index
    edge_weight = batch.edge_weight
    blk        = model.block

    def one_pass():
        xin = x_input_of(batch)
        with tm("time_emb"):
            t_vec = model.time_emb(t)
        with tm("beta_emb"):
            beta_vec = model.beta_emb(inv_temp * model.beta_scale)
        with tm("damp_proj -> alpha_node"):
            alpha_node = torch.sigmoid(model.damp_proj(beta_vec))[batch_vec]
        with tm("field stats + t_vec add"):
            t_vec = t_vec + beta_vec
            field = xin[:, -1:]
            h_signed_mean = batch.field_mean.unsqueeze(-1)
            h_std = batch.field_std.unsqueeze(-1)
        with tm("precompute_cond (t_proj) + tanh/clamp"):
            gamma, beta_c = blk.precompute_cond(t_vec, batch_vec)
            gamma = torch.tanh(gamma)
            beta_c = beta_c.clamp(-2, 2)
        with tm("in_lin_proj + h_proj"):
            h_proj = model.h_proj(field)
            x = model.in_lin_proj(xin) + h_proj

        with tm(f"recurrent loop x{NUM_ITERS} (total)"):
            for _ in range(NUM_ITERS):
                x = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per)
                x = x + alpha_node * blk(x, edge_index, batch_vec,
                                         gamma, beta_c, edge_weight, num_graphs) + h_proj
                x_mean = x.view(num_graphs, N_per, -1).mean(dim=1)[batch_vec]
                x = x + torch.sigmoid(model.global_gate) * x_mean

        with tm("output act+concat+out_lin_proj"):
            x = model.act(x)
            x = torch.cat((x, h_signed_mean[batch_vec], h_std[batch_vec],
                           (h_std * inv_temp.unsqueeze(-1))[batch_vec]), dim=-1)
            out = model.out_lin_proj(x)
        if backward:
            with tm("backward"):
                loss = out.float().pow(2).mean()
                loss.backward()
                model.zero_grad(set_to_none=True)
        return out

    for _ in range(WARMUP):
        one_pass()
    tm.totals.clear(); tm.counts.clear(); tm.order.clear()
    for _ in range(REPS):
        one_pass()
    tm.report(REPS, f"(1) COMPONENT BREAKDOWN  [{'fwd+bwd' if backward else 'fwd'}]"
                     f"  N={N_per}  B={num_graphs}")


# ════════════════════════════════════════════════════════════════════════════
#  2. PER-ITERATION BREAKDOWN  (inside the recurrent loop)
# ════════════════════════════════════════════════════════════════════════════
def per_iter_breakdown(model, batch, t):
    tm = SectionTimer()
    inv_temp   = batch.inv_temp
    num_graphs = batch.num_graphs
    batch_vec  = batch.batch
    N_per      = batch.num_nodes // num_graphs
    edge_index = batch.edge_index
    edge_weight = batch.edge_weight
    blk        = model.block

    xin = x_input_of(batch)
    with torch.no_grad():
        t_vec = model.time_emb(t) + model.beta_emb(inv_temp * model.beta_scale)
        alpha_node = torch.sigmoid(model.damp_proj(model.beta_emb(inv_temp * model.beta_scale)))[batch_vec]
        gamma, beta_c = blk.precompute_cond(t_vec, batch_vec)
        gamma = torch.tanh(gamma); beta_c = beta_c.clamp(-2, 2)
        field = xin[:, -1:]
        h_proj = model.h_proj(field)
        x0 = model.in_lin_proj(xin) + h_proj
        gate = torch.sigmoid(model.global_gate)

    def one_iter_series():
        x = x0
        for _ in range(NUM_ITERS):
            with tm("rms_norm_fixed_N"):
                xn = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per)
            with tm("FiLM affine + SiLU"):
                xm = xn * (1.0 + gamma) + beta_c
                xm = blk.act(xm)
            with tm("MLP residual (2 Linear+SiLU)"):
                xm = xm + blk.mlp(xm)
            with tm("GraphConv (PyG scatter/gather)"):
                xc = blk.conv0(xm, edge_index, edge_weight)
            with tm("residual add (alpha*conv+h_proj)"):
                x = xn + alpha_node * xc + h_proj
            with tm("global mean (dense view+mean+gather)"):
                x_mean = x.view(num_graphs, N_per, -1).mean(dim=1)[batch_vec]
            with tm("gated add"):
                x = x + gate * x_mean

    with torch.no_grad():
        for _ in range(WARMUP):
            one_iter_series()
        tm.totals.clear(); tm.counts.clear(); tm.order.clear()
        for _ in range(REPS):
            one_iter_series()
    tm.report(REPS * NUM_ITERS,
              f"(2) PER-ITERATION BREAKDOWN (per single loop iter)  N={N_per}  B={num_graphs}")

    # GraphConv vs dense-matmul fraction
    gc = tm.totals["GraphConv (PyG scatter/gather)"]
    dense = (tm.totals["MLP residual (2 Linear+SiLU)"]
             + tm.totals["FiLM affine + SiLU"]
             + tm.totals["rms_norm_fixed_N"]
             + tm.totals["global mean (dense view+mean+gather)"]
             + tm.totals["gated add"]
             + tm.totals["residual add (alpha*conv+h_proj)"])
    tot = gc + dense
    print(f"  >> GraphConv        : {100*gc/tot:5.1f}% of loop compute")
    print(f"  >> dense/pointwise  : {100*dense/tot:5.1f}% of loop compute")


# ════════════════════════════════════════════════════════════════════════════
#  3. TORCH.PROFILER  (top ops by cuda_time_total)
# ════════════════════════════════════════════════════════════════════════════
def torch_profiler(model, batch, t, backward=False, tag=""):
    activities = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def step():
        xin = x_input_of(batch)
        out = model(xin, batch, t)
        if backward:
            loss = out.float().pow(2).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        step()
    sync()

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(10):
            with record_function("full_pass"):
                step()
            prof.step()
    sync()

    sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
    print(f"\n(3) TORCH.PROFILER top ops by {sort_key}  [{tag}]")
    print("=" * 100)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=22))

    # kernel-launch / op-count summary
    evts = prof.key_averages()
    total_calls = sum(e.count for e in evts)
    gc_calls = sum(e.count for e in evts
                   if any(k in e.key.lower() for k in
                          ("scatter", "gather", "index_select", "spmm", "segment")))
    print(f"  >> total profiled op invocations (10 passes): {total_calls}  "
          f"(~{total_calls/10:.0f}/pass)")
    print(f"  >> scatter/gather/index_select ops         : {gc_calls} "
          f"(~{gc_calls/10:.0f}/pass)")


# ════════════════════════════════════════════════════════════════════════════
#  4. PEAK GPU MEMORY
# ════════════════════════════════════════════════════════════════════════════
def peak_memory(model, batch, t, backward):
    if DEVICE.type != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 1024**2
    xin = x_input_of(batch)
    out = model(xin, batch, t)
    if backward:
        loss = out.float().pow(2).mean()
        loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    model.zero_grad(set_to_none=True)
    print(f"  [{'fwd+bwd' if backward else 'fwd'}] peak {peak:8.1f} MB "
          f"(baseline {base:6.1f} MB, activations+grads {peak-base:8.1f} MB)")


# ════════════════════════════════════════════════════════════════════════════
#  5. N-SCALING SWEEP  (sparse vs dense crossover)
# ════════════════════════════════════════════════════════════════════════════
def scaling_sweep(model):
    print("\n" + "=" * 78)
    print("(5) N-SCALING SWEEP  (total nodes B*N held ~constant to isolate per-graph N)")
    print("=" * 78)
    print(f"{'N':>7}{'B':>7}{'E/graph':>9}{'fwd ms':>10}{'fwd+bwd ms':>12}"
          f"{'GConv %':>9}{'peakMB(bwd)':>13}")
    print("-" * 78)
    for N in (144, 1024, 4096, 10000):
        B = max(1, TOTAL_NODES // N)
        ei = build_rrg(N, DEGREE)
        batch = make_batch(B, ei, N)
        t = torch.randint(0, TIMESTEPS - 1, (B,), device=DEVICE).float()

        def fwd():
            with torch.no_grad():
                model(x_input_of(batch), batch, t)

        def fwdbwd():
            out = model(x_input_of(batch), batch, t)
            out.float().pow(2).mean().backward()
            model.zero_grad(set_to_none=True)

        ms_f = time_fn(fwd, warmup=10, reps=30)
        ms_fb = time_fn(fwdbwd, warmup=10, reps=30)

        # GraphConv share of forward at this N (isolate conv vs whole loop)
        gc_share = graphconv_share(model, batch, t)

        torch.cuda.reset_peak_memory_stats()
        fwdbwd()
        sync()
        peak = torch.cuda.max_memory_allocated() / 1024**2

        print(f"{N:>7}{B:>7}{ei.shape[1]//1:>9}{ms_f:>10.3f}{ms_fb:>12.3f}"
              f"{gc_share:>8.1f}%{peak:>13.1f}")
        del batch
        torch.cuda.empty_cache()


def graphconv_share(model, batch, t):
    """Fraction of a single loop iteration spent in GraphConv (sync-timed)."""
    inv_temp = batch.inv_temp
    num_graphs = batch.num_graphs
    batch_vec = batch.batch
    N_per = batch.num_nodes // num_graphs
    ei, ew = batch.edge_index, batch.edge_weight
    blk = model.block
    xin = x_input_of(batch)
    with torch.no_grad():
        t_vec = model.time_emb(t) + model.beta_emb(inv_temp * model.beta_scale)
        gamma, beta_c = blk.precompute_cond(t_vec, batch_vec)
        gamma = torch.tanh(gamma); beta_c = beta_c.clamp(-2, 2)
        x = model.in_lin_proj(xin) + model.h_proj(xin[:, -1:])

        def loop_all():
            xn = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per)
            xm = blk.act(xn * (1 + gamma) + beta_c)
            xm = xm + blk.mlp(xm)
            xc = blk.conv0(xm, ei, ew)
            _ = xn + xc

        def conv_only():
            xn = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per)
            xm = blk.act(xn * (1 + gamma) + beta_c)
            xm = xm + blk.mlp(xm)
            _ = blk.conv0(xm, ei, ew)

        t_all = time_fn(loop_all, warmup=10, reps=40)
        # crude: time conv alone by timing (all) - (all minus conv) via mlp-only
        def no_conv():
            xn = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per)
            xm = blk.act(xn * (1 + gamma) + beta_c)
            _ = xm + blk.mlp(xm)
        t_noconv = time_fn(no_conv, warmup=10, reps=40)
    return max(0.0, 100.0 * (t_all - t_noconv) / t_all)


# ════════════════════════════════════════════════════════════════════════════
#  6. torch.compile graph breaks + speedup
# ════════════════════════════════════════════════════════════════════════════
def compile_probe(batch, t):
    print("\n" + "=" * 78)
    print("(6) torch.compile PROBE (graph breaks + speedup)")
    print("=" * 78)
    model = build_model()
    xin = x_input_of(batch)

    # count graph breaks
    try:
        import torch._dynamo as dynamo
        expl = dynamo.explain(model)(xin, batch, t)
        print(f"  graph breaks : {expl.graph_break_count}")
        print(f"  graphs       : {expl.graph_count}")
        if expl.break_reasons:
            seen = set()
            for r in expl.break_reasons:
                reason = getattr(r, "reason", str(r))
                if reason not in seen:
                    seen.add(reason)
                    print(f"    - {reason}")
    except Exception as e:
        print(f"  [dynamo.explain failed: {type(e).__name__}: {e}]")

    def eager():
        with torch.no_grad():
            model(xin, batch, t)
    ms_eager = time_fn(eager, warmup=10, reps=40)

    try:
        cmodel = torch.compile(model, dynamic=False)
        def compiled():
            with torch.no_grad():
                cmodel(xin, batch, t)
        ms_comp = time_fn(compiled, warmup=20, reps=40)
        print(f"  eager fwd    : {ms_eager:.3f} ms")
        print(f"  compiled fwd : {ms_comp:.3f} ms   ({ms_eager/ms_comp:.2f}x)")
    except Exception as e:
        print(f"  [torch.compile failed: {type(e).__name__}: {e}]")
        print(f"  eager fwd    : {ms_eager:.3f} ms")


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="all|component|iter|prof|mem|scale|compile")
    args = ap.parse_args()

    print(f"Device       : {DEVICE}  "
          f"({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else ''})")
    print(f"torch        : {torch.__version__}")
    print(f"Config       : N={N_TRAIN} B={B_TRAIN} base_ch={BASE_CH} "
          f"num_iters={NUM_ITERS} time_emb={TIME_EMB} degree={DEGREE}")

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params : {n_params:,}")

    ei = build_rrg(N_TRAIN, DEGREE)
    batch = make_batch(B_TRAIN, ei, N_TRAIN)
    t = torch.randint(0, TIMESTEPS - 1, (B_TRAIN,), device=DEVICE).float()
    print(f"Batch        : {batch.num_nodes} nodes, {batch.num_edges} edges "
          f"(E/graph={ei.shape[1]}, avg deg={ei.shape[1]/N_TRAIN:.2f})")

    if args.only in ("all", "component"):
        component_breakdown(model, batch, t, backward=False)
        component_breakdown(model, batch, t, backward=True)
    if args.only in ("all", "iter"):
        per_iter_breakdown(model, batch, t)
    if args.only in ("all", "prof"):
        torch_profiler(model, batch, t, backward=False, tag="fwd")
        torch_profiler(model, batch, t, backward=True, tag="fwd+bwd")
    if args.only in ("all", "mem"):
        print("\n(4) PEAK GPU MEMORY (training config)")
        peak_memory(model, batch, t, backward=False)
        peak_memory(model, batch, t, backward=True)
    if args.only in ("all", "scale"):
        scaling_sweep(model)
    if args.only in ("all", "compile"):
        compile_probe(batch, t)

    print("\nDone.")


if __name__ == "__main__":
    main()
