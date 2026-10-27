"""
Targeted profiler for unetGnn_cfg.py — measures wall-clock cost of every
sub-component in GNNUnet.forward() under realistic training settings.

Settings mirror train_cfg.py defaults:
  L=12 (N=144 nodes), batch_size=1024, base_ch=64, num_iters=10, time_emb_dim=32
"""
import sys, os, time, contextlib
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.profiler import profile, record_function, ProfilerActivity

from models.unetGnn_cfg import GNNUnet, rms_norm_fixed_N

# ── config ────────────────────────────────────────────────────────────────────
BASE_CH     = 64
NUM_ITERS   = 10
TIME_EMB    = 32
L           = 12
N           = L * L        # 144
BATCH_SIZE  = 1024
TIMESTEPS   = 1000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP      = 5
REPEATS     = 20           # timing repetitions (each = one forward pass)

print(f"Device : {DEVICE}")
print(f"N={N}, B={BATCH_SIZE}, base_ch={BASE_CH}, num_iters={NUM_ITERS}\n")

# ── helpers ───────────────────────────────────────────────────────────────────
def grid_edge_index(L):
    src, dst = [], []
    for i in range(L):
        for j in range(L):
            node = i * L + j
            for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                ni, nj = (i+di)%L, (j+dj)%L
                src.append(node); dst.append(ni*L+nj)
    return torch.tensor([src, dst], dtype=torch.long)

edge_index_single = grid_edge_index(L)   # [2, 4*N]

def make_batch(bs):
    """Mimics train_cfg.py's dataset.get() structure."""
    graphs = []
    for _ in range(bs):
        x          = torch.randint(0, 2, (N, 1)).float() * 2 - 1
        field      = torch.randn(N, 1) * 0.5
        mk         = torch.randn(N, 1) * 0.5
        beta       = torch.rand(1) * 0.5 + 0.2          # random inv_temp in [0.2, 0.7]
        field_mean = field.mean().view(1)
        field_std  = field.std().view(1).clamp(min=1e-6)

        # Degree-normalised edge weights (same as dataset.get)
        src, dst = edge_index_single
        deg = torch.zeros(N).scatter_add_(0, dst, torch.ones(src.shape[0]))
        deg_inv_sqrt  = deg.pow(-0.5)
        deg_norm      = (deg_inv_sqrt[src] * deg_inv_sqrt[dst]).contiguous()
        edge_weight   = (beta * deg_norm).contiguous()

        graphs.append(Data(
            x=x, mk=mk, field=field,
            field_mean=field_mean, field_std=field_std,
            edge_index=edge_index_single.clone(),
            edge_weight=edge_weight,
            inv_temp=beta,
            instance_idx=torch.tensor(0),
        ))
    return Batch.from_data_list(graphs).to(DEVICE)

def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

# ── model ─────────────────────────────────────────────────────────────────────
model = GNNUnet(base_ch=BASE_CH, time_emb_dim=TIME_EMB,
                discrete=True, num_iters=NUM_ITERS, damping=0.3).to(DEVICE)
model.eval()  # forward-only profiling (no grad)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Pre-build a fixed batch and time vector (avoid re-building every rep)
batch = make_batch(BATCH_SIZE)
t     = torch.randint(0, TIMESTEPS-1, (BATCH_SIZE,), device=DEVICE).float()

# ── warm-up ───────────────────────────────────────────────────────────────────
print("Warming up ...")
with torch.no_grad():
    for _ in range(WARMUP):
        x_input = torch.cat([batch.x, batch.field], dim=1)  # [N_total, 2]
        _ = model(x_input, batch, t, drop_beta=False)
sync()
print("Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  COMPONENT-LEVEL WALL-CLOCK TIMER (manual instrumentation)
# ─────────────────────────────────────────────────────────────────────────────
class Timer:
    def __init__(self):
        self.totals = {}
        self.counts = {}

    @contextlib.contextmanager
    def __call__(self, name):
        sync()
        t0 = time.perf_counter()
        yield
        sync()
        dt = (time.perf_counter() - t0) * 1e3  # ms
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, reps):
        grand = sum(self.totals.values())
        print(f"\n{'─'*64}")
        print(f"{'Component':<40} {'ms/fwd':>8}  {'%':>6}")
        print(f"{'─'*64}")
        for k, v in self.totals.items():
            avg = v / reps
            pct = 100.0 * v / grand
            print(f"  {k:<38} {avg:8.3f}  {pct:6.1f}%")
        print(f"{'─'*64}")
        print(f"  {'TOTAL (wall)':<38} {grand/reps:8.3f}  100.0%")
        print(f"{'─'*64}\n")

timer = Timer()

print(f"Running {REPEATS} forward passes (component-level timing) ...")
with torch.no_grad():
    for rep in range(REPEATS):
        x_input = torch.cat([batch.x, batch.field], dim=1)   # [N_total, 2]
        inv_temp   = batch.inv_temp
        batch_vec  = batch.batch
        num_graphs = batch.num_graphs
        num_nodes  = batch.num_nodes
        edge_index = batch.edge_index
        edge_weight= batch.edge_weight

        # ── time embedding ────────────────────────────────────────────────
        with timer("time_emb"):
            t_vec = model.time_emb(t)

        # ── beta embedding ────────────────────────────────────────────────
        with timer("beta_emb"):
            beta_vec = model.beta_emb(inv_temp * model.beta_scale)

        with timer("damp_proj"):
            alpha = torch.sigmoid(model.damp_proj(beta_vec))

        # ── field stats ───────────────────────────────────────────────────
        with timer("field_stats_lookup"):
            field       = x_input[:, -1].unsqueeze(-1)
            h_signed_mean = batch.field_mean.unsqueeze(-1)
            h_std         = batch.field_std.unsqueeze(-1)
            t_vec_comb    = t_vec + beta_vec
            alpha_node    = alpha[batch_vec]

        # ── precompute conditioning ───────────────────────────────────────
        with timer("precompute_cond (t_proj)"):
            gamma, beta_cond = model.block.precompute_cond(t_vec_comb, batch_vec)

        # ── initial projections ───────────────────────────────────────────
        with timer("in_lin_proj + h_proj"):
            h_proj = model.h_proj(field)
            x = model.in_lin_proj(x_input) + h_proj

        # ── recurrent iterations ──────────────────────────────────────────
        with timer(f"loop x{NUM_ITERS} (total)"):
            for _ in range(NUM_ITERS):
                x = rms_norm_fixed_N(x.contiguous(), num_graphs, num_nodes // num_graphs)
                x = x + alpha_node * model.block(x, edge_index, batch_vec,
                                                  gamma, beta_cond, edge_weight, num_graphs) + h_proj

        # ── output ────────────────────────────────────────────────────────
        with timer("output_act + concat + out_lin_proj"):
            x = model.act(x)
            x = torch.cat((x,
                            h_signed_mean[batch_vec],
                            h_std[batch_vec],
                            (h_std * inv_temp.unsqueeze(-1))[batch_vec]), dim=-1)
            _ = model.out_lin_proj(x)

timer.report(REPEATS)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-ITERATION BREAKDOWN inside the loop
# ─────────────────────────────────────────────────────────────────────────────
print("Per-iteration breakdown (single forward pass, each op timed over loop) ...")
loop_timer = Timer()
x_input = torch.cat([batch.x, batch.field], dim=1)
field   = x_input[:, -1].unsqueeze(-1)

with torch.no_grad():
    t_vec    = model.time_emb(t)
    beta_vec = model.beta_emb(inv_temp * model.beta_scale)
    alpha    = torch.sigmoid(model.damp_proj(beta_vec))
    t_vec   += beta_vec
    gamma, beta_cond = model.block.precompute_cond(t_vec, batch_vec)
    h_proj   = model.h_proj(field)
    x        = model.in_lin_proj(x_input) + h_proj
    alpha_node = alpha[batch_vec]
    blk      = model.block

    for rep in range(REPEATS):
        x_iter = x.clone()
        for _ in range(NUM_ITERS):
            with loop_timer("  rms_norm_fixed_N"):
                x_iter = rms_norm_fixed_N(x_iter.contiguous(), num_graphs, num_nodes // num_graphs)

            with loop_timer("  GNNLayer: FiLM (tanh+clamp)"):
                x_mod = x_iter * (1.0 + blk.tanh(gamma)) + beta_cond.clamp(-2, 2)
                x_mod = blk.act(x_mod)

            with loop_timer("  GNNLayer: MLP"):
                x_mlp = blk.mlp(x_mod)
                x_mod = x_mod + x_mlp

            with loop_timer("  GNNLayer: GraphConv"):
                x_conv = blk.conv0(x_mod, edge_index, edge_weight)

            with loop_timer("  residual + h_proj add"):
                x_iter = x_iter + alpha_node * x_conv + h_proj

loop_timer.report(REPEATS * NUM_ITERS)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TORCH PROFILER (top ops by CUDA/CPU time)
# ─────────────────────────────────────────────────────────────────────────────
print("Running torch.profiler (5 steps) ...")
PROF_DIR = os.path.join(os.path.dirname(__file__), "prof_results_cfg")
os.makedirs(PROF_DIR, exist_ok=True)

activities = [ProfilerActivity.CPU]
if DEVICE.type == "cuda":
    activities.append(ProfilerActivity.CUDA)

with torch.no_grad():
    with profile(activities=activities, record_shapes=True,
                 with_stack=True, profile_memory=True) as prof:
        for step in range(5):
            x_input = torch.cat([batch.x, batch.field], dim=1)
            with record_function("forward"):
                _ = model(x_input, batch, t, drop_beta=False)
            prof.step()

sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
print(f"\nTOP 25 OPS (by {sort_key})")
print("="*80)
print(prof.key_averages(group_by_stack_n=3).table(sort_by=sort_key, row_limit=25))

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MEMORY FOOTPRINT
# ─────────────────────────────────────────────────────────────────────────────
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        x_input = torch.cat([batch.x, batch.field], dim=1)
        _ = model(x_input, batch, t, drop_beta=False)
    torch.cuda.synchronize()
    peak_mb  = torch.cuda.max_memory_allocated() / 1024**2
    alloc_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"\nGPU memory — peak: {peak_mb:.1f} MB,  currently allocated: {alloc_mb:.1f} MB")

print(f"\nTensorBoard trace: {PROF_DIR}")
print("  tensorboard --logdir", PROF_DIR)
