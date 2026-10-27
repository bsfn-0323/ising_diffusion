import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from .base import DiffusionProcess
class D3PMProcess(DiffusionProcess):
    def __init__(self, betas: torch.Tensor, device: torch.device, lambda_aux: float = 0.01, lambda_mk: float = 0.1, p_cfg: float = 0.0,
                 lambda_fa: float = 0.0, fa_estimator: str = "unbiased",
                 uniform_N: bool = True):
        self.device = device
        self.timesteps = len(betas) # Inherits whatever schedule length you pass
        self.K = 2
        self.lambda_aux = lambda_aux
        self.lambda_mk = lambda_mk

        # How compute_loss reduces the per-node terms:
        #   True  -> flat mean over every node in the batch (original path).
        #            Correct and cheapest when all graphs have the same N.
        #   False -> reduce per graph, then over graphs, so each sample gets
        #            equal weight no matter its size.  Required for batches that
        #            mix system sizes.
        # The two agree algebraically at fixed N, so leaving this True keeps
        # existing runs bit-for-bit identical.  Overridable per call.
        self.uniform_N = uniform_N

        # --- Flip-asymmetry ("free-energy") loss L_A (OFF by default) ---
        # Supervises the model's flip log-density ratio r_theta(s)=log p(s)/p(-s)
        # toward the exact Boltzmann tilt r*(s)=2*beta*(h.s), which fixes the
        # relative Z2 mode weights without conditioning on m_k. lambda_fa==0 =>
        # this machinery never runs and behaviour is identical to before.
        self.lambda_fa = lambda_fa
        self.fa_estimator = fa_estimator      # "unbiased" (2-sample) or "single" (1-sample, biased)

        self.p_cfg = p_cfg
        self.Q_t = torch.zeros((self.timesteps, self.K, self.K), device=device)
        for t in range(self.timesteps):
            b = betas[t] # Applies your custom schedule step-by-step
            self.Q_t[t] = torch.tensor([
                [1.0 - b/2, b/2],
                [b/2, 1.0 - b/2]
            ], device=device)

        self.Q_bar_t = torch.zeros_like(self.Q_t)
        self.Q_bar_t[0] = self.Q_t[0]
        for t in range(1, self.timesteps):
            self.Q_bar_t[t] = torch.matmul(self.Q_bar_t[t-1], self.Q_t[t])

    def _spin_to_idx(self, x: torch.Tensor) -> torch.Tensor:
        """Maps physical spins {-1, 1} to indices {0, 1}."""
        return ((x + 1) / 2).long().squeeze()

    def _idx_to_spin(self, idx: torch.Tensor) -> torch.Tensor:
        """Maps indices {0, 1} back to physical spins {-1, 1}."""
        return (idx * 2 - 1).float().unsqueeze(-1)
    
    def _vb_kl_node(self, model, x0_idx, xt_idx, xt_physical, t, batch,
                    use_mk: bool = False, include_L0: bool = False, drop_beta: bool = False,
                    detach_body: bool = False,use_both=False):
        """Per-node D3PM VLB KL term for a GIVEN (x0, x_t, t), plus the model logits.

        Factored out of compute_loss so that both the ordinary loss and the
        flip-asymmetry loss evaluate the identical VB math. Returns
        (loss_vb_node [N], logits [N]).  The caller supplies x_t (indices +
        physical), so the flip-asymmetry loss can pass the shared flipped noise.

        detach_body: when True, logits = body_logit.detach() + tilt_b, so a loss
        built on these logits trains ONLY the tilt head (the correlation body is
        frozen).  Used by flip_asymmetry_loss so L_A cannot distort the energy
        channel.  Requires the model to have a tilt head (use_tilt_head=True).
        """
        N = len(x0_idx)
        idx_arange = torch.arange(N, device=self.device)

        # Model prediction p_theta(x_0 | x_t).  Only the spin column (col 0) is
        # x_t; the conditioning column (col -1) is the field h (or m_k).
        # print(use_mk,flush=True)
        if use_mk:
            # print("Using mk")
            x_input = torch.cat([xt_physical, batch.mk], dim=1)
        else:
            field_zero = torch.zeros_like(batch.field)
            # print("Not using mk")
            # x_input = torch.cat([xt_physical, batch.field], dim=1)
            x_input = torch.cat([xt_physical, field_zero], dim=1)
        if use_both:
            x_input = torch.cat([xt_physical, batch.field,batch.mk], dim=1)

        if detach_body:
            body, tilt_b = model(x_input, batch, t.float(), drop_beta=drop_beta, return_parts=True)
            logits = body.detach().float().squeeze(-1) + tilt_b.float().squeeze(-1)
        else:
            logits = model(x_input, batch, t.float(), drop_beta=drop_beta)
            logits = logits.float().squeeze(-1)
        p_plus = torch.sigmoid(logits)
        p_theta_x0 = torch.stack([1.0 - p_plus, p_plus], dim=-1)  # (N, 2)

        # Forward marginal q(x_t | x_0) for this x0 (needed in the posterior denom).
        q_bar_t_batch = self.Q_bar_t[t[batch.batch]]
        p_xt = q_bar_t_batch[idx_arange, x0_idx]

        # Variational Lower Bound posterior KL, per node.
        t_prev = torch.clamp(t - 1, min=0)
        q_t_batch = self.Q_t[t[batch.batch]]
        q_bar_t_prev_batch = self.Q_bar_t[t_prev[batch.batch]]

        denom = p_xt.gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)

        qt_trans = q_t_batch.transpose(1, 2)[idx_arange, xt_idx]
        qbar_prev_trans = q_bar_t_prev_batch[idx_arange, x0_idx]

        q_posterior = (qt_trans * qbar_prev_trans) / denom

        p_theta_xt_prev = torch.zeros_like(p_theta_x0)
        for possible_x0 in range(self.K):
            test_p_xt = q_bar_t_batch[:, possible_x0, :]
            test_denom = test_p_xt.gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)

            test_qbar_prev = q_bar_t_prev_batch[:, possible_x0, :]
            test_posterior = (qt_trans * test_qbar_prev) / test_denom

            p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)

        loss_vb_node = F.kl_div(torch.log(p_theta_xt_prev.clamp(min=1e-6)), q_posterior, reduction='none').sum(dim=-1)

        # t == 0 (i.e. x_1 -> x_0) carries no KL posterior. By default we drop it;
        # if include_L0 is set we substitute the L_0 reconstruction term
        # -log p_theta(x_0 | x_1) so the full variational bound of eq. (1) is computed.
        is_t0 = (t[batch.batch] == 0)
        if include_L0:
            l0_node = -torch.log(p_theta_x0[idx_arange, x0_idx].clamp(min=1e-6))
            loss_vb_node = torch.where(is_t0, l0_node, loss_vb_node)
        else:
            loss_vb_node = torch.where(is_t0, torch.zeros_like(loss_vb_node), loss_vb_node)

        return loss_vb_node, logits

    def compute_loss(self, model: torch.nn.Module, batch, t: torch.Tensor,reduction = 'mean', include_L0: bool = False, use_mk:bool = False,use_both:bool=False,
                     uniform_N: bool = None) -> tuple[torch.Tensor, torch.Tensor]:
        x0_idx = self._spin_to_idx(batch.x)
        N = len(x0_idx)
        idx_arange = torch.arange(N, device=self.device)

        # 1. Forward Sample q(x_t | x_0)
        q_bar_t_batch = self.Q_bar_t[t[batch.batch]]
        p_xt = q_bar_t_batch[idx_arange, x0_idx]
        xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze()
        xt_physical = self._idx_to_spin(xt_idx)

        # 2. Model prediction + VB (shared with the flip-asymmetry loss).
        drop_beta = (self.p_cfg > 0.0) and (torch.rand(1).item() < self.p_cfg)
        loss_vb_node, logits = self._vb_kl_node(
            model, x0_idx, xt_idx, xt_physical, t, batch,
            use_mk=use_mk, include_L0=include_L0, drop_beta=drop_beta,use_both=use_both)

        p_plus = torch.sigmoid(logits)

        # 3. Reduction.  A flat mean over every node in the batch weights each
        # graph by its size, so in a batch that MIXES system sizes a 5000-node
        # sample carries 50x the gradient of a 100-node one -- and since t is
        # drawn per graph, that big graph's single t dominates the step too.
        # uniform_N=False reduces per graph first (global_*_pool), giving every
        # sample equal weight regardless of N.  At fixed N the two are
        # algebraically identical, so uniform_N=True (default) leaves existing
        # runs untouched.
        if uniform_N is None:
            uniform_N = self.uniform_N

        if uniform_N:
            # 3a. Auxiliary Loss (using numerically stable BCE)
            loss_aux = F.binary_cross_entropy_with_logits(logits, x0_idx.float())
            if reduction == 'mean':
                loss_vb = loss_vb_node.mean()
            elif reduction == 'sum':
                loss_vb = loss_vb_node.sum()
        else:
            bvec = batch.batch
            B    = batch.num_graphs

            loss_aux_node = F.binary_cross_entropy_with_logits(
                logits, x0_idx.float(), reduction='none')                    # [N]
            loss_aux = global_mean_pool(loss_aux_node.unsqueeze(-1), bvec, size=B).mean()

            if reduction == 'mean':
                # per-graph node-mean, then mean over graphs
                loss_vb = global_mean_pool(loss_vb_node.unsqueeze(-1), bvec, size=B).mean()
            elif reduction == 'sum':
                # per-graph VLB (extensive, summed over that graph's nodes),
                # then mean over graphs.  Differs from the uniform_N 'sum'
                # branch by a factor num_graphs; nothing in the repo calls
                # compute_loss with reduction='sum'.
                loss_vb = global_add_pool(loss_vb_node.unsqueeze(-1), bvec, size=B).mean()
        # x0_pred_idx = (logits > 0.0).long()
        p_plus = torch.sigmoid(logits)
        # s_expected = (2.0 * p_plus - 1.0).unsqueeze(-1)

        # E[s_i | x_t, J, h] ≈ <s_i>*(1 - λ_t) + x_{0,i}*λ_t  where
        # λ_t = Q_bar_t[0,0] - Q_bar_t[0,1] is the signal retention factor
        # (1 at t=0, 0 at t=T).  Weight loss_mk by (1 - λ_t): vanishes when
        # s_expected tracks the specific sample (low t, conflicts with
        # loss_aux), full weight when s_expected reliably estimates <s_i>
        # (high t, no x_t contamination).  More principled than t/T since it
        # uses the actual schedule rather than a linear proxy.
        # diffusion_factor = (q_bar_t_batch[:, 0, 0] - q_bar_t_batch[:, 0, 1]).unsqueeze(-1)
        # mk_weight = 1.0 - diffusion_factor
        # loss_mk = (F.mse_loss(s_expected, batch.mk, reduction='none') * mk_weight).mean()

        total_loss = loss_vb + self.lambda_aux * loss_aux

        # return total_loss, self._idx_to_spin(x0_pred_idx)
        return total_loss, p_plus

    def flip_asymmetry_loss(self, model, batch, use_mk: bool = False, return_diag: bool = False):
        r"""Flip-asymmetry ("free-energy") loss L_A.

        Enforces the exact Z2 tilt of the Boltzmann distribution so the model
        reproduces the relative +m/-m mode weights, *without* conditioning on m_k.

        For the RFIM H(s) = -1/2 s.J.s - h.s, a global flip s -> -s leaves the
        coupling term invariant and flips the field term, giving the exact
        log-density ratio (the target):

            r*(s) = log[p(s)/p(-s)] = 2 * beta * (h . s).                 (per graph)

        The model's own ratio is the endpoint (t=0) log-density difference,
        estimated by the flip difference of the D3PM VLB (sigma(t,T)-free because
        it is the SUM over t, not a per-t marginal):

            r_theta(s) ~ (T-1) * [ sum_i K_t(-s,-x_t)_i - sum_i K_t(s,x_t)_i ]

        evaluated at a shared t with shared flipped noise (x_t^flip = -x_t) to
        cancel variance and the VLB gap.  L_A = E_s[(r_theta - r*)^2].

        Estimator (self.fa_estimator):
          "unbiased": two independent draws, (r1 - tau)*(r2 - tau)  (E = (r-tau)^2).
          "single":   one draw, (r1 - tau)^2  (adds a theta-dependent Var term -> biased).

        All arithmetic is done in fp32 (the (T-1) factor and the product would
        overflow/underflow fp16); the model forward may still run under autocast.
        """
        bvec = batch.batch
        B = batch.num_graphs
        Tm1 = float(self.timesteps - 1)              # scaling: t ~ U{0,...,T-2}

        x0_phys = batch.x                            # [N,1] physical spins +-1
        x0_idx = self._spin_to_idx(batch.x)          # [N]
        x0_idx_flip = self._spin_to_idx(-batch.x)    # [N] flipped clean config
        N = len(x0_idx)
        idx_arange = torch.arange(N, device=self.device)

        # --- exact target tau_g = 2 * beta_g * sum_{i in g} h_i s_i (fp32) ---
        hs_node = (batch.field * x0_phys).squeeze(-1).float()         # [N]
        hs_graph = torch.zeros(B, device=self.device).scatter_add_(0, bvec, hs_node)
        tau = 2.0 * batch.inv_temp.float() * hs_graph                 # [B]

        def _rhat():
            # one draw of the flip log-ratio estimate r_theta, per graph
            t = torch.randint(0, self.timesteps - 1, (B,), device=self.device)
            q_bar = self.Q_bar_t[t[bvec]]
            p_xt = q_bar[idx_arange, x0_idx]
            xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze()
            xt_phys = self._idx_to_spin(xt_idx)

            # real: (x0, x_t) ; flip: (-x0, -x_t), SAME field/beta, spins flipped.
            # detach_body=True -> L_A trains ONLY the tilt head; the correlation
            # body is frozen w.r.t. L_A, so the energy channel is protected.
            vb_real, _ = self._vb_kl_node(model, x0_idx, xt_idx, xt_phys, t, batch,
                                          use_mk=use_mk, include_L0=True, drop_beta=False,
                                          detach_body=True)
            xt_idx_flip = self._spin_to_idx(-xt_phys)
            vb_flip, _ = self._vb_kl_node(model, x0_idx_flip, xt_idx_flip, -xt_phys, t, batch,
                                          use_mk=use_mk, include_L0=True, drop_beta=False,
                                          detach_body=True)

            K_real = torch.zeros(B, device=self.device).scatter_add_(0, bvec, vb_real.float())
            K_flip = torch.zeros(B, device=self.device).scatter_add_(0, bvec, vb_flip.float())
            return Tm1 * (K_flip - K_real)           # [B] estimate of r_theta(s)

        r1 = _rhat()
        if self.fa_estimator == "single":
            L = (r1 - tau) ** 2
        else:  # "unbiased": second independent draw makes the square unbiased
            r2 = _rhat()
            L = (r1 - tau) * (r2 - tau)
        loss = L.mean()

        if return_diag:
            # Diagnostics for the --fa_probe pass: r_hat vs tau scale/correlation
            # and the raw per-t |K_flip - K_real| magnitude.
            with torch.no_grad():
                a = r1 - r1.mean(); b = tau - tau.mean()
                corr = (a * b).sum() / (a.norm() * b.norm() + 1e-12)
                diag = {
                    "fa_mean_rhat": r1.mean().item(),
                    "fa_mean_tau": tau.mean().item(),
                    "fa_corr": corr.item(),
                    "fa_mean_absKdiff": (r1 / Tm1).abs().mean().item(),
                }
            return loss, diag
        return loss

    def _predict_logits(self, model, x_t, t, batch, use_cfg=False, guidance_scale=1.0,
                        symmetrize=False, use_mk=False,use_both = False):
        """Per-node x0 logits, optionally CFG-guided and/or Z2-symmetrized.

        Z2 symmetrization enforces the exact physical constraint
        logit(-x, -h) = -logit(x, h) by returning 0.5*(f(x,h) - f(-x,-h)).
        The flip negates both columns of x_input (spins and per-node field)
        AND batch.field_mean, since the model reads the global field mean from
        the batch (h_std is even, so left untouched). Removes the spurious mode
        bias that otherwise tilts the +m/-m weight. Costs a second forward pass.
        """
        def _cond(x_in, b):
            if use_cfg and guidance_scale != 1.0:
                lc = model(x_in, b, t.float(), drop_beta=False).float().squeeze(-1)
                lu = model(x_in, b, t.float(), drop_beta=True).float().squeeze(-1)
                return lu + guidance_scale * (lc - lu)
            return model(x_in, b, t.float(), drop_beta=False).float().squeeze(-1)
        if use_mk:
            print("Using mk",flush=True)
            x_input = torch.cat([x_t, batch.mk], dim=1)
        else:
            print("Not using mk",flush=True)
            x_input = torch.cat([x_t, batch.field], dim=1)
        if use_both:
            x_input = torch.cat([x_t, batch.field,batch.mk], dim=1)
        logits = _cond(x_input, batch)
        if symmetrize:
            orig_fm = batch.field_mean
            batch.field_mean = -orig_fm
            logits_flip = _cond(-x_input, batch)
            batch.field_mean = orig_fm            # restore in place
            logits = 0.5 * (logits - logits_flip)
        return logits

    def d3pm_step(self, model, x_t, t, batch, use_cfg: bool = False, guidance_scale: float = 1.0,
                  symmetrize: bool = False):
        xt_idx = self._spin_to_idx(x_t)
        N = len(xt_idx)
        idx_arange = torch.arange(N, device=self.device)

        logits = self._predict_logits(model, x_t, t, batch, use_cfg, guidance_scale, symmetrize)

        p_plus = torch.sigmoid(logits)
        p_theta_x0 = torch.stack([1.0 - p_plus, p_plus], dim=-1)
        
        t_val = t[0]
        if t_val == 0:
            # Sample x_0 ~ Categorical(p_theta(x_0|x_1)) rather than argmax, so the
            # final transition is a proper stochastic kernel whose density matches
            # backward_path_log_prob (needed for a valid acceptance ratio).
            x0_idx = torch.multinomial(p_theta_x0, num_samples=1).squeeze()
            return self._idx_to_spin(x0_idx)

        q_t_batch = self.Q_t[t_val].unsqueeze(0).expand(N, -1, -1)
        q_bar_prev_batch = self.Q_bar_t[t_val - 1].unsqueeze(0).expand(N, -1, -1)
        q_bar_t_batch = self.Q_bar_t[t_val].unsqueeze(0).expand(N, -1, -1)
        
        p_theta_xt_prev = torch.zeros_like(p_theta_x0)
        qt_trans = q_t_batch.transpose(1, 2)[idx_arange, xt_idx]
        
        for possible_x0 in range(self.K):
            test_denom = q_bar_t_batch[:, possible_x0, :].gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
            test_qbar_prev = q_bar_prev_batch[:, possible_x0, :]
            
            test_posterior = (qt_trans * test_qbar_prev) / test_denom
            p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)
            
        xt_prev_idx = torch.multinomial(p_theta_xt_prev, num_samples=1).squeeze()
        return self._idx_to_spin(xt_prev_idx)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        """Forward diffuses clean fixed spins x_0 to time t."""
        x0_idx = self._spin_to_idx(x_0)
        idx_arange = torch.arange(len(x0_idx), device=self.device)
        
        # Retrieve the cumulative transition matrices for the batch
        q_bar_t_batch = self.Q_bar_t[t[batch_vec]]
        
        # Get transition probabilities for the specific clean state
        p_xt = q_bar_t_batch[idx_arange, x0_idx]
        
        # Sample the noisy fixed state
        xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze()
        return self._idx_to_spin(xt_idx)

    def d3pm_strided_step(self, model, x_t, t_curr, t_prev, batch, use_cfg: bool = False,
                          guidance_scale: float = 1.0, symmetrize: bool = False,use_mk=False,use_both=False):
        """Reverse step that jumps from t_curr to t_prev (t_prev < t_curr).

        Uses the multi-step transition matrix Q[t_prev → t_curr] derived from
        the cumulative Q_bar matrices, so the posterior is exact under the model.
        When t_prev < 0 or t_curr == 0, decodes greedily.
        """
        # print(use_mk)
        xt_idx = self._spin_to_idx(x_t)
        N = len(xt_idx)
        idx_arange = torch.arange(N, device=self.device)

        logits = self._predict_logits(model, x_t, t_curr, batch, use_cfg, guidance_scale, symmetrize,use_mk=use_mk,use_both=use_both)

        p_plus = torch.sigmoid(logits)
        p_theta_x0 = torch.stack([1.0 - p_plus, p_plus], dim=-1)

        t_curr_val = t_curr[0].item()
        t_prev_val = t_prev[0].item()

        if t_curr_val == 0 or t_prev_val < 0:
            # Sample x_0 ~ Categorical(p_theta(x_0|x_t)) rather than argmax, so the
            # final transition is a proper stochastic kernel whose density matches
            # backward_path_log_prob (needed for a valid acceptance ratio).
            x0_idx = torch.multinomial(p_theta_x0, num_samples=1).squeeze()
            return self._idx_to_spin(x0_idx)

        # Multi-step transition Q[t_prev → t_curr] for the symmetric 2-state chain:
        # staying prob p = (a_curr + a_prev - 1) / (2*a_prev - 1)
        a_curr = self.Q_bar_t[t_curr_val][0, 0]
        a_prev = self.Q_bar_t[t_prev_val][0, 0]
        p_stay = ((a_curr + a_prev - 1.0) / (2.0 * a_prev - 1.0).clamp(min=1e-8)).clamp(0.0, 1.0)

        Q_stride = torch.tensor([
            [p_stay,        1.0 - p_stay],
            [1.0 - p_stay,  p_stay      ],
        ], device=self.device)

        q_bar_prev_batch = self.Q_bar_t[t_prev_val].unsqueeze(0).expand(N, -1, -1)
        q_bar_t_batch    = self.Q_bar_t[t_curr_val].unsqueeze(0).expand(N, -1, -1)
        q_stride_batch   = Q_stride.unsqueeze(0).expand(N, -1, -1)

        p_theta_xt_prev = torch.zeros_like(p_theta_x0)
        qt_trans = q_stride_batch.transpose(1, 2)[idx_arange, xt_idx]  # [N, 2]

        for possible_x0 in range(self.K):
            test_denom     = q_bar_t_batch[:, possible_x0, :].gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
            test_qbar_prev = q_bar_prev_batch[:, possible_x0, :]
            test_posterior = (qt_trans * test_qbar_prev) / test_denom
            p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)

        xt_prev_idx = torch.multinomial(p_theta_xt_prev, num_samples=1).squeeze()
        return self._idx_to_spin(xt_prev_idx)

    def forward_path_log_prob(
        self,
        trajectory: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log probability of a (possibly strided) trajectory under the forward process q.

        trajectory: (T, P, N) physical spins {-1, 1}, ordered clean→noisy.
                    trajectory[0] is x_0 (the clean state).
        timesteps:  LongTensor length T. Set timesteps[0] = -1 to mark the clean
                    state; remaining entries are valid indices into Q_bar_t.

        Returns (P,) log-probabilities summed over N.
        """
        T, P, N = trajectory.shape
        log_prob = torch.zeros(P, N, device=self.device)

        for i in range(1, T):
            t_prev = timesteps[i - 1].item()
            t_curr = timesteps[i].item()
            x_prev_idx = ((trajectory[i - 1] + 1) / 2).long()  # (P, N)
            x_curr_idx = ((trajectory[i]     + 1) / 2).long()  # (P, N)

            if t_prev < 0:
                # clean x_0 → first noisy state: q(x_t | x_0) = Q_bar_t[x_0, x_t]
                trans_probs = self.Q_bar_t[t_curr][x_prev_idx, x_curr_idx]  # (P, N)
            else:
                # multi-step forward kernel via symmetric 2-state closed form
                a_curr = self.Q_bar_t[t_curr][0, 0]
                a_prev = self.Q_bar_t[t_prev][0, 0]
                p_stay = ((a_curr + a_prev - 1.0) / (2.0 * a_prev - 1.0).clamp(min=1e-8)).clamp(0.0, 1.0)
                same = (x_curr_idx == x_prev_idx).float()  # (P, N)
                trans_probs = same * p_stay + (1.0 - same) * (1.0 - p_stay)  # (P, N)

            log_prob += torch.log(trans_probs.clamp(min=1e-6))

        return log_prob.sum(dim=-1)  # (P,)

    def backward_path_log_prob(
        self,
        model: torch.nn.Module,
        trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """
        Log probability of a trajectory under the learned reverse process p_theta.

        trajectory: (T, P, N) physical spins {-1, 1}, same clean→noisy ordering
                    as forward_path_log_prob.
        timesteps:  LongTensor length T. Set timesteps[0] = -1 for the clean state.
        batch:      PyG batch with P copies of the graph (batch.num_graphs == P),
                    e.g. Batch.from_data_list([data] * P).

        Iterates trajectory[-1] (noisy) → trajectory[0] (clean), evaluating
        p_theta(x_{t_prev} | x_t) at each observed state.
        Returns (P,) log-probabilities summed over N.
        """
        T, P, N = trajectory.shape
        log_prob = torch.zeros(P, N)  # CPU — accumulation never touches GPU

        # pre-fetch only the needed 2×2 Q_bar slices to CPU (negligible size)
        q_bar_cpu = {}
        for t in timesteps:
            t_val = t.item()
            if t_val >= 0 and t_val not in q_bar_cpu:
                q_bar_cpu[t_val] = self.Q_bar_t[t_val].cpu()

        with torch.no_grad():  # no activation graph — biggest memory saving
            for i in range(T - 2, -1, -1):
                t_prev_val = timesteps[i].item()
                t_curr_val = timesteps[i + 1].item()

                # only the current x_t slice goes to GPU (model needs it + batch.field)
                x_t          = trajectory[i + 1].to(self.device)         # (P, N) GPU
                x_t_idx      = ((trajectory[i + 1] + 1) / 2).long().cpu()  # (P, N) CPU
                x_t_prev_idx = ((trajectory[i]     + 1) / 2).long().cpu()  # (P, N) CPU

                x_input  = torch.cat([x_t.reshape(P * N, 1), batch.field], dim=1)
                t_tensor = torch.full((P,), t_curr_val, dtype=torch.float, device=self.device)
                p_plus   = torch.sigmoid(
                    model(x_input, batch, t_tensor, drop_beta=False).float().squeeze(-1).reshape(P, N)
                ).cpu()                                                    # (P, N) → CPU immediately
                del x_t  # release GPU slice

                if t_prev_val < 0:
                    prob = torch.where(x_t_prev_idx == 1, p_plus, 1.0 - p_plus)  # (P, N) CPU
                else:
                    Qbt = q_bar_cpu[t_curr_val]   # (2, 2) CPU
                    Qbp = q_bar_cpu[t_prev_val]   # (2, 2) CPU

                    a_curr, a_prev = Qbt[0, 0].item(), Qbp[0, 0].item()
                    p_stay = max(0.0, min(1.0, (a_curr + a_prev - 1.0) / max(2.0 * a_prev - 1.0, 1e-8)))

                    same      = (x_t_idx == x_t_prev_idx).float()
                    qt_factor = same * p_stay + (1.0 - same) * (1.0 - p_stay)  # (P, N) CPU

                    qbp0 = Qbp[0, x_t_prev_idx]            # (P, N) CPU
                    qbt0 = Qbt[0, x_t_idx].clamp(min=1e-6)
                    qbp1 = Qbp[1, x_t_prev_idx]
                    qbt1 = Qbt[1, x_t_idx].clamp(min=1e-6)

                    weight = (qbp0 / qbt0) * (1.0 - p_plus) + (qbp1 / qbt1) * p_plus
                    prob   = qt_factor * weight                           # (P, N) CPU

                log_prob += torch.log(prob.clamp(min=1e-6))

        return log_prob.sum(dim=-1)  # (P,) CPU

    # ------------------------------------------------------------------
    # Helpers for diffusion-proposal MCMC (see mcmc.py).
    #
    # These give a SELF-CONSISTENT pair: sample_reverse_trajectory (proposal
    # x' + cached log p_theta path) and score_reverse_trajectory (log p_theta
    # path of a given trajectory, e.g. the forward-lifted current state), both
    # built from the single normalized kernel reverse_step_dist so the proposed
    # and current states are scored under an identical density. sample_forward_
    # trajectory lifts a clean state by the forward process on the same grid.
    #
    # Trajectory format matches forward_path_log_prob / backward_path_log_prob:
    #   trajectory : (S+1, P, N) physical spins, ordered clean -> noisy,
    #                trajectory[0] = x_0 (clean).
    #   t_vals     : LongTensor length S+1, t_vals[0] = -1 (clean flag), the
    #                remaining entries are strictly increasing indices into Q_bar_t.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def strided_forward_step(self, x, t_prev, t_curr):
        """Sample x_{t_curr} ~ q(x_{t_curr} | x_{t_prev}) on the strided grid.

        x: (P, N) physical spins at time t_prev (t_prev < 0 => clean x_0).
        Uses the same symmetric 2-state kernel as forward_path_log_prob.
        """
        x_idx = ((x + 1) / 2).long()  # (P, N)
        if t_prev < 0:
            probs = self.Q_bar_t[t_curr][x_idx]  # (P, N, 2) = q(x_t | x_0)
        else:
            a_curr = self.Q_bar_t[t_curr][0, 0]
            a_prev = self.Q_bar_t[t_prev][0, 0]
            p_stay = ((a_curr + a_prev - 1.0) / (2.0 * a_prev - 1.0).clamp(min=1e-8)).clamp(0.0, 1.0)
            probs = torch.empty(*x_idx.shape, self.K, device=self.device)
            probs[..., 0] = torch.where(x_idx == 0, p_stay, 1.0 - p_stay)
            probs[..., 1] = torch.where(x_idx == 1, p_stay, 1.0 - p_stay)
        idx = torch.multinomial(probs.reshape(-1, self.K), 1).reshape(x_idx.shape)
        return (idx * 2 - 1).float()

    @torch.no_grad()
    def sample_forward_trajectory(self, x0, t_vals):
        """Lift a clean state x0 (P, N) to a forward trajectory on grid t_vals.

        Returns (trajectory (S+1, P, N), log_f (P,)), where log_f = log q(x_{1:T}|x_0)
        is evaluated with the verified forward_path_log_prob for consistency.
        """
        P, N = x0.shape
        S1 = len(t_vals)
        traj = torch.empty(S1, P, N, device=self.device)
        traj[0] = x0
        for i in range(1, S1):
            traj[i] = self.strided_forward_step(traj[i - 1], int(t_vals[i - 1]), int(t_vals[i]))
        log_f = self.forward_path_log_prob(traj, t_vals)
        return traj, log_f

    @torch.no_grad()
    def reverse_step_dist(self, model, x_t, batch, t_curr, t_prev):
        """Normalized per-node reverse kernel p_theta(x_{t_prev} | x_t): (P, N, K).

        Same posterior math as backward_path_log_prob / d3pm_strided_step, but
        returns the full normalized distribution so it can be used both to sample
        (proposal) and to score (current state) with one identical density.
        """
        P, N = x_t.shape
        x_flat = x_t.reshape(P * N, 1)
        t_tensor = torch.full((P,), t_curr, device=self.device, dtype=torch.long)
        logits = self._predict_logits(model, x_flat, t_tensor, batch).reshape(P, N)
        p_plus = torch.sigmoid(logits)
        p_theta_x0 = torch.stack([1.0 - p_plus, p_plus], dim=-1)  # (P, N, K)

        if t_prev < 0 or t_curr == 0:
            return p_theta_x0  # final step: p_theta(x_0 | x_t)

        a_curr = self.Q_bar_t[t_curr][0, 0]
        a_prev = self.Q_bar_t[t_prev][0, 0]
        p_stay = ((a_curr + a_prev - 1.0) / (2.0 * a_prev - 1.0).clamp(min=1e-8)).clamp(0.0, 1.0)

        xt_idx = ((x_t + 1) / 2).long()          # (P, N)
        Qbt, Qbp = self.Q_bar_t[t_curr], self.Q_bar_t[t_prev]
        qbt0 = Qbt[0, xt_idx].clamp(min=1e-6)    # (P, N)
        qbt1 = Qbt[1, xt_idx].clamp(min=1e-6)

        probs = torch.empty(P, N, self.K, device=self.device)
        for j in range(self.K):
            qt_factor = torch.where(xt_idx == j, p_stay, 1.0 - p_stay)
            weight = (Qbp[0, j] / qbt0) * (1.0 - p_plus) + (Qbp[1, j] / qbt1) * p_plus
            probs[..., j] = qt_factor * weight
        probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-12)  # -> true density
        return probs

    @torch.no_grad()
    def sample_reverse_trajectory(self, model, batch, t_vals):
        """Generate a proposal trajectory X' ~ reverse process on grid t_vals.

        Returns (trajectory (S+1, P, N), log_b (P,)) where log_b = log p_theta(x_{0:T-1}|x_T)
        is accumulated during generation (cached) -- no second model pass needed.
        """
        P = batch.num_graphs
        N = batch.num_nodes // P
        S1 = len(t_vals)
        traj = torch.empty(S1, P, N, device=self.device)
        x = (torch.randint(0, self.K, (P, N), device=self.device) * 2 - 1).float()  # prior at t_vals[-1]
        traj[S1 - 1] = x
        log_b = torch.zeros(P, N, device=self.device)
        for k in range(S1 - 1, 0, -1):
            d = self.reverse_step_dist(model, x, batch, int(t_vals[k]), int(t_vals[k - 1]))
            idx = torch.multinomial(d.reshape(-1, self.K), 1).reshape(P, N)
            log_b += torch.log(d.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))
            x = (idx * 2 - 1).float()
            traj[k - 1] = x
        return traj, log_b.sum(-1)

    @torch.no_grad()
    def score_reverse_trajectory(self, model, traj, t_vals, batch):
        """log p_theta(x_{0:T-1} | x_T) of a given trajectory under the same
        normalized kernel used by sample_reverse_trajectory. Returns (P,)."""
        S1, P, N = traj.shape
        log_b = torch.zeros(P, N, device=self.device)
        for k in range(S1 - 1, 0, -1):
            d = self.reverse_step_dist(model, traj[k], batch, int(t_vals[k]), int(t_vals[k - 1]))
            idx = ((traj[k - 1] + 1) / 2).long()
            log_b += torch.log(d.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))
        return log_b.sum(-1)

    def ddpm_step(self, *args, **kwargs):
        raise NotImplementedError("For discrete processes, use d3pm_step instead of ddpm_step.")

    def ddim_step(self, *args, **kwargs):
        raise NotImplementedError("DDIM skipping is not mathematically supported for discrete Markov chains.")
