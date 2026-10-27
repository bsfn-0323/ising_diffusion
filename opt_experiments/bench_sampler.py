"""
Validate + benchmark the CUDA-graphed D3PM sampler against the eager sampler,
using the REAL trained checkpoint (base_ch=128, num_iters=12, ~83k params).

Correctness (rigorous, before any timing is trusted):
  A. faithfulness: _generic_step_body(mode=full) == stock process.d3pm_step for a
     t>0 step, seed-matched -> identical sampled spins (proves the copied math).
  B. graph == eager: det-mode graph replay probs == eager body probs (max|Δ|~1e-6).
  C. distribution: full-mode graph vs stock, many draws from a fixed (x_t,t) ->
     per-node mean magnetization agrees within sampling noise.
Then times full 1000-step sample_d3pm eager vs graphed across batch sizes.

NOTE: beta_scale is not stored in the checkpoint; we use the current default
(100.0). If training used a different value the samples are physically off, but
the graph-vs-eager EQUIVALENCE we validate here is unaffected.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.unetGnn_cfg import GNNUnet
from processes.sde import get_cosine_schedule
from processes.discrete import D3PMProcess
from profile_unetgnn_cfg import build_rrg, DEVICE
from opt_experiments.graphed_sampler import (GraphedIsingSampler, CUDAGraphedStep,
                                             _generic_step_body)

CKPT_DIR = "../rfim_learn_fields/last_pls/D3PM_CFG/2026-06-24/hna_L12_lr0.002"
N = 144
TIMESTEPS = 1000
torch.manual_seed(0)


def load_model():
    cfg_path = os.path.join(CKPT_DIR, "config.json")
    ckpt = os.path.join(CKPT_DIR, "model_last.pt")
    if os.path.exists(cfg_path) and os.path.exists(ckpt):
        a = json.load(open(cfg_path))["architecture"]
        m = GNNUnet(base_ch=a["base_ch"], time_emb_dim=a["time_emb_dim"],
                    discrete=True, num_iters=a["num_iters"])
        r = m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True),
                              strict=False)
        print(f"Loaded {ckpt}  (base_ch={a['base_ch']} num_iters={a['num_iters']} "
              f"params={sum(p.numel() for p in m.parameters())}; "
              f"ignored unexpected={r.unexpected_keys})")
        return m.to(DEVICE).eval()
    from profile_unetgnn_cfg import build_model
    print("Checkpoint not found — using untrained model.")
    return build_model().eval()


model = load_model()
betas = get_cosine_schedule(TIMESTEPS)
process = D3PMProcess(betas=betas, device=DEVICE)
edge_index_single = build_rrg(N, 3)
field_single = torch.randn(N, 1)
inv_temp_single = torch.tensor(0.4)

sampler = GraphedIsingSampler(model=model, process=process, device=DEVICE, accelerator=None)


def make_state(B, t_val):
    x, batch = sampler._prepare_batch_v3(B, edge_index_single, field_single, inv_temp_single)
    t = torch.full((B,), t_val, device=DEVICE, dtype=torch.long)
    return x, batch, t


print("\n" + "=" * 70)
print("CORRECTNESS")
print("=" * 70)

# A. faithfulness of the copied t>0 body vs stock d3pm_step (seed-matched)
B = 8; x, batch, t = make_state(B, 500)
torch.manual_seed(123); s_stock = process.d3pm_step(model, x, t, batch)
torch.manual_seed(123); s_body = _generic_step_body(process, model, x, t, batch, mode="full")
print(f"A. body == stock d3pm_step (seed-matched, t=500): {torch.equal(s_stock, s_body)}  "
      f"[max|Δ|={(s_stock-s_body).abs().max().item():.2e}]")

# B. det-mode graph replay == eager body probs, over several t
stepd = CUDAGraphedStep(process, model, batch, x, t, mode="det")
maxd = 0.0
for tv in (900, 500, 100, 1):
    xr, _, _ = make_state(B, tv)
    p_graph = stepd.run(xr, tv)
    p_eager = _generic_step_body(process, model, xr,
                                 torch.full((B,), tv, device=DEVICE, dtype=torch.long),
                                 batch, mode="det")
    maxd = max(maxd, (p_graph - p_eager).abs().max().item())
print(f"B. det graph replay == eager probs (t=900..1): max|Δ|={maxd:.2e}")

# C. distribution match: full-graph vs stock, per-node mean over many draws
tv = 300; x0, batch, t = make_state(64, tv)
stepf = CUDAGraphedStep(process, model, batch, x0, t, mode="full")
dg, de = [], []
for _ in range(200):
    dg.append(stepf.run(x0, tv).view(64, N, 1))
    de.append(process.d3pm_step(model, x0, t, batch).view(64, N, 1))
mg = torch.cat(dg, 0).mean(0).squeeze()
me = torch.cat(de, 0).mean(0).squeeze()
print(f"C. full-graph vs stock per-node mean magnetization (t=300, 200x64 draws): "
      f"max|Δ|={(mg-me).abs().max().item():.3f}  (want < ~0.05 sampling noise)")

print("\n" + "=" * 70)
print("TIMING  (full 1000-step sample_d3pm: eager vs graphed[full])")
print("=" * 70)
print(f"{'B':>5}{'eager s':>11}{'graphed s':>11}{'speedup':>10}"
      f"{'eager ms/step':>16}{'graphed ms/step':>18}")
for B in (1, 4, 16, 64):
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    _ = sampler.sample_d3pm(B, edge_index_single, field_single, inv_temp_single)
    torch.cuda.synchronize(); te = time.perf_counter() - t0
    _ = sampler.sample_d3pm_graphed(B, edge_index_single, field_single, inv_temp_single, mode="full")
    tg = sampler._last_time
    print(f"{B:>5}{te:>11.3f}{tg:>11.3f}{te/tg:>9.2f}x"
          f"{te/TIMESTEPS*1e3:>16.3f}{tg/TIMESTEPS*1e3:>18.3f}")

print("\nDone.")
