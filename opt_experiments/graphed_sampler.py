"""
CUDA-graphed D3PM sampler — captures the FULL generic (t>0) reverse step
(GNN forward + posterior math + multinomial) into a single CUDA graph and
replays it 999 times, running only the final t==0 step eagerly.

Source untouched: GraphedIsingSampler subclasses engines.samplers.IsingSampler
and reuses _prepare_batch_v3 + the eager process.d3pm_step for t==0.

Why replicate the t>0 step body here instead of calling process.d3pm_step:
the stock d3pm_step has `if t_val == 0:` (processes/discrete.py:156) which forces
a .item() host sync — illegal during graph capture. `_generic_step_body` below is
a branch-free, host-sync-free copy of the t>0 path (discrete.py:146-178). Q
matrices are indexed by the 0-dim tensor t[0] (a gather, capturable), never by a
python int, so a single graph is correct for every t>0 when we copy the step
index into the static `t` buffer before each replay.

mode="full" : capture through multinomial (graph-safe RNG). ~7x, not seed-reproducible.
mode="det"  : capture only the deterministic posterior probs; multinomial runs
              eager afterwards. Seed-reproducible -> used to PROVE correctness.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from engines.samplers import IsingSampler


def _generic_step_body(process, model, x_t, t, batch, mode):
    """Branch-free copy of d3pm_step's t>0 path. Returns spins (full) or probs (det)."""
    K = process.K
    dev = x_t.device
    xt_idx = ((x_t + 1) / 2).long().squeeze()          # _spin_to_idx
    N = xt_idx.shape[0]
    idx_arange = torch.arange(N, device=dev)

    # model forward (plain path: no cfg / no symmetrize)
    x_input = torch.cat([x_t, batch.field], dim=1)
    logits = model(x_input, batch, t.float(), drop_beta=False).float().squeeze(-1)
    p_plus = torch.sigmoid(logits)
    p_theta_x0 = torch.stack([1.0 - p_plus, p_plus], dim=-1)

    # index_select with a 1-elem GPU index tensor is a gather (capturable);
    # Q_t[t[0]] with a 0-dim tensor would force a host .item() sync -> illegal
    # during graph capture. tv1-1 >= 0 always here (t==0 handled eagerly).
    tv1 = t[:1]                                          # [1] long, on device
    q_t_batch      = process.Q_t.index_select(0, tv1).expand(N, -1, -1)
    q_bar_prev     = process.Q_bar_t.index_select(0, tv1 - 1).expand(N, -1, -1)
    q_bar_t_batch  = process.Q_bar_t.index_select(0, tv1).expand(N, -1, -1)

    p_theta_xt_prev = torch.zeros_like(p_theta_x0)
    qt_trans = q_t_batch.transpose(1, 2)[idx_arange, xt_idx]
    for possible_x0 in range(K):
        test_denom = q_bar_t_batch[:, possible_x0, :].gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
        test_qbar_prev = q_bar_prev[:, possible_x0, :]
        test_posterior = (qt_trans * test_qbar_prev) / test_denom
        p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)

    if mode == "det":
        return p_theta_xt_prev
    xt_prev_idx = torch.multinomial(p_theta_xt_prev, num_samples=1).squeeze()
    return (xt_prev_idx * 2 - 1).float().unsqueeze(-1)   # _idx_to_spin


class CUDAGraphedStep:
    """Captures _generic_step_body into a CUDA graph over static (x_t, t) buffers."""
    def __init__(self, process, model, batch, sample_x, sample_t, mode="full"):
        self.mode = mode
        self.static_x = sample_x.clone()
        self.static_t = sample_t.clone()

        def body():
            return _generic_step_body(process, model, self.static_x, self.static_t, batch, mode)

        # warm up on a side stream so all workspaces/autotune settle before capture
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.static_out = body()

    def run(self, x_t, t_index: int):
        self.static_x.copy_(x_t)
        self.static_t.fill_(t_index)
        self.graph.replay()
        return self.static_out.clone()


class GraphedIsingSampler(IsingSampler):
    @torch.no_grad()
    def sample_d3pm_graphed(self, batch_size, edge_index_single, field_single,
                            inv_temp_single, mode="full"):
        x, batch = self._prepare_batch_v3(batch_size, edge_index_single, field_single, inv_temp_single)
        T = self.process.timesteps

        sample_t = torch.full((batch_size,), T - 1, device=self.device, dtype=torch.long)
        step = CUDAGraphedStep(self.process, self.model, batch, x, sample_t, mode=mode)

        start = time.perf_counter()
        for i in tqdm(range(T - 1, 0, -1), desc=f"D3PM graphed [{mode}]", leave=False):
            out = step.run(x, i)
            if mode == "det":
                x = self.process._idx_to_spin(torch.multinomial(out, 1).squeeze())
            else:
                x = out
        # final t==0 step: eager (stock branch handles x0 decode)
        t0 = torch.full((batch_size,), 0, device=self.device, dtype=torch.long)
        x = self.process.d3pm_step(model=self.model, x_t=x, t=t0, batch=batch)
        torch.cuda.synchronize()
        self._last_time = time.perf_counter() - start
        return x.view(batch_size, -1, 1)
