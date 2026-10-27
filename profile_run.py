"""
Profile D3PM training forward+backward pass using standard argparse settings:
  L=12 (N=144), batch_size=1024, base_ch=64, num_iters=10, time_emb_dim=32
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from processes.sde import get_cosine_schedule
from processes.discrete import D3PMProcess
from models.unetGnn import GNNUnet

# ── standard settings ─────────────────────────────────────────────────────────
TIMESTEPS   = 1000
BASE_CH     = 64
NUM_ITERS   = 10
TIME_EMB    = 32
L           = 12
N           = L * L       # 144 nodes per graph
BATCH_SIZE  = 1024        # number of graphs
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── build model & process ─────────────────────────────────────────────────────
betas   = get_cosine_schedule(TIMESTEPS).to(DEVICE)
process = D3PMProcess(betas=betas, device=DEVICE, lambda_aux=0.1, lambda_mk=0)
model   = GNNUnet(base_ch=BASE_CH, time_emb_dim=TIME_EMB,
                  discrete=True, num_iters=NUM_ITERS, damping=0.3).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)

# ── build a 2D grid (square lattice) edge_index for an L×L graph ──────────────
def grid_edge_index(L):
    src, dst = [], []
    for i in range(L):
        for j in range(L):
            node = i * L + j
            for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                ni, nj = (i+di) % L, (j+dj) % L
                src.append(node); dst.append(ni*L+nj)
    return torch.tensor([src, dst], dtype=torch.long)

edge_index_single = grid_edge_index(L)   # [2, 4*N]

# ── build a fake batch ─────────────────────────────────────────────────────────
def make_batch(bs):
    graphs = []
    for _ in range(bs):
        x     = torch.randint(0, 2, (N, 1)).float() * 2 - 1   # {-1,+1}
        field = torch.randn(N, 1) * 0.5
        mk    = torch.randn(N, 1) * 0.5
        graphs.append(Data(x=x, field=field, mk=mk,
                           edge_index=edge_index_single.clone(),
                           instance_idx=torch.tensor(0)))
    return Batch.from_data_list(graphs).to(DEVICE)

# ── warm-up ───────────────────────────────────────────────────────────────────
print("Warming up...")
batch = make_batch(BATCH_SIZE)
for _ in range(3):
    t = torch.randint(0, TIMESTEPS-1, (BATCH_SIZE,), device=DEVICE)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = process.compute_loss(model, batch, t)
    loss.backward()
    optimizer.step()

if DEVICE.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

# ── profile ───────────────────────────────────────────────────────────────────
print("Profiling...")
PROF_DIR = os.path.join(os.path.dirname(__file__), "prof_results")
os.makedirs(PROF_DIR, exist_ok=True)

with profile(
    activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE.type == "cuda" else []),
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=tensorboard_trace_handler(PROF_DIR),
) as prof:
    for step in range(5):
        with record_function("make_batch"):
            batch = make_batch(BATCH_SIZE)
            t     = torch.randint(0, TIMESTEPS-1, (BATCH_SIZE,), device=DEVICE)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with record_function("compute_loss"):
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                loss, _ = process.compute_loss(model, batch, t)

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer_step"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        prof.step()

# ── print top ops by CUDA time (or CPU if no GPU) ─────────────────────────────
sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
print("\n" + "="*80)
print(f"TOP 30 OPS (by {sort_key})")
print("="*80)
print(prof.key_averages(group_by_stack_n=5).table(
    sort_by=sort_key, row_limit=30))

# ── per-section breakdown ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("SECTION BREAKDOWN (custom record_function labels)")
print("="*80)
for evt in prof.key_averages():
    if evt.key in ("make_batch","compute_loss","backward","optimizer_step"):
        cuda_t = getattr(evt, "self_cuda_time_total", None) or getattr(evt, "cuda_time_total", 0)
        cpu_t  = getattr(evt, "self_cpu_time_total",  None) or getattr(evt, "cpu_time_total",  0)
        t_ms = (cuda_t if DEVICE.type == "cuda" else cpu_t) / 1e3 / 5
        print(f"  {evt.key:20s}  {t_ms:8.2f} ms/step")

if DEVICE.type == "cuda":
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\nPeak GPU memory:  {peak_mb:.1f} MB")

print(f"\nTensorBoard trace saved to: {PROF_DIR}")
print("View with: tensorboard --logdir", PROF_DIR)
