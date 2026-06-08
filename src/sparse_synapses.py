"""
sparse_synapses.py — Edge-list connectivity backend for BrainNet
================================================================
A drop-in replacement for the dense `SynapseMatrix` that stores synapses as an
edge list (pre, post, weight) instead of a dense [N, N] matrix. Memory and
per-step cost scale with the number of *synapses* E, not N², which is what lets
the network grow past the dense ceiling (an 8 GB GPU holds only a few thousand
neurons densely, but millions of sparse edges).

The dynamics are identical to the dense backend — propagation is a scatter-add
instead of a matmul, and STDP / STP / homeostasis / pruning act per-edge. Enable
with `NeuronParams(sparse_synapses=True)`.

Note: the growth cone's pairwise spike-train correlation is still O(N²); that is
the next thing to sparsify (candidate sampling). The per-step hot path — the
dominant cost in long runs — is fully O(E) here.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import torch

if TYPE_CHECKING:
    from core import NeuronParams, NeuromodulatorState


class SparseSynapses:
    """
    Edge-list synapse population. Public interface matches `SynapseMatrix`:
        propagate, apply_stdp, apply_stp, homeostatic_scale, prune,
        add_synapse, add_synapses_bulk, num_synapses(_in/_out), weight_stats,
        state_dict / load_state_dict, and a `mask` property (materialized).
    """

    def __init__(self, N: int, params: "NeuronParams", device: torch.device):
        self.N      = N
        self.params = params
        self.device = device

        # Edge list (initially empty)
        self.pre = torch.empty(0, dtype=torch.long,  device=device)
        self.post = torch.empty(0, dtype=torch.long, device=device)
        self.w   = torch.empty(0, dtype=torch.float, device=device)

        # Per-neuron STDP eligibility traces
        self.trace_pre  = torch.zeros(N, device=device)
        self.trace_post = torch.zeros(N, device=device)

        # Per-axon short-term plasticity state (Tsodyks–Markram)
        self.stp_u = torch.full((N,), params.U_stp, device=device)
        self.stp_x = torch.ones(N, device=device)

        # Per-edge eligibility trace (three-factor reward/stress learning).
        # Kept aligned with the edge arrays; zero unless enable_reward_learning.
        self.elig = torch.empty(0, dtype=torch.float, device=device)

    # ── construction ──────────────────────────────────────────────
    def _lin(self, pre, post):
        return pre * self.N + post

    def add_synapses_bulk(self, pre: torch.Tensor, post: torch.Tensor,
                          weights: Optional[torch.Tensor] = None):
        """Add many edges at once, de-duplicating against self-loops, repeats,
        and any edges that already exist."""
        pre  = pre.to(self.device).long()
        post = post.to(self.device).long()
        keep = pre != post
        pre, post = pre[keep], post[keep]
        if weights is None:
            weights = torch.empty(pre.shape, device=self.device).uniform_(0.01, 0.1)
        else:
            weights = weights.to(self.device)[keep]

        # de-duplicate new pairs (keep first occurrence)
        lin = self._lin(pre, post)
        lin_u, first = np.unique(lin.cpu().numpy(), return_index=True)
        first = torch.from_numpy(first).to(self.device)
        pre, post, weights = pre[first], post[first], weights[first]
        lin = lin[first]

        # drop pairs that already exist
        if self.pre.numel() > 0:
            existing = set(self._lin(self.pre, self.post).cpu().numpy().tolist())
            keep2 = torch.tensor([int(x) not in existing for x in lin.cpu().numpy()],
                                 dtype=torch.bool, device=self.device)
            pre, post, weights = pre[keep2], post[keep2], weights[keep2]

        self.pre  = torch.cat([self.pre, pre])
        self.post = torch.cat([self.post, post])
        self.w    = torch.cat([self.w, weights])
        self.elig = torch.cat([self.elig, torch.zeros(pre.numel(), device=self.device)])

    def add_synapse(self, pre: int, post: int, weight: Optional[float] = None):
        """Add a single edge (callers must ensure it does not already exist —
        the growth cone only proposes non-existing pairs)."""
        if pre == post:
            return
        if weight is None:
            weight = float(torch.empty(1).uniform_(0.01, 0.1))
        self.pre  = torch.cat([self.pre,  torch.tensor([pre],  device=self.device)])
        self.post = torch.cat([self.post, torch.tensor([post], device=self.device)])
        self.w    = torch.cat([self.w,    torch.tensor([weight], device=self.device)])
        self.elig = torch.cat([self.elig, torch.zeros(1, device=self.device)])

    # ── hot path ──────────────────────────────────────────────────
    def propagate(self, spike_f: torch.Tensor) -> torch.Tensor:
        """I_post[j] = Σ_{edges i→j} spike_f[i] · w   (scatter-add)."""
        I_post = torch.zeros(self.N, device=self.device)
        if self.w.numel() == 0:
            return I_post
        contrib = spike_f[self.pre] * self.w
        return I_post.index_add_(0, self.post, contrib)

    def apply_stdp(self, spikes: torch.Tensor, dt: float,
                   modulator: "NeuromodulatorState"):
        """Per-edge pair STDP — identical rule to the dense backend."""
        p = self.params
        dt_ms = dt * 1000.0
        self.trace_pre  = self.trace_pre  * float(np.exp(-dt_ms / p.tau_plus))
        self.trace_post = self.trace_post * float(np.exp(-dt_ms / p.tau_minus))

        if self.w.numel() > 0:
            # LTP: post fires now, weighted by recent pre activity.
            ltp = self.trace_pre[self.pre] * spikes[self.post] * p.A_plus
            # LTD: pre fires now, weighted by recent post activity.
            ltd = spikes[self.pre] * self.trace_post[self.post] * p.A_minus
            if p.enable_reward_learning:
                # Tag causally-active edges (LTP-shaped); dopamine/cortisol
                # convert the tag to potentiation/depression later.
                self.elig = self.elig * float(np.exp(-dt_ms / p.tau_elig)) + ltp
            else:
                self.w = (self.w + (ltp * modulator.dopamine - ltd)).clamp(p.w_min, p.w_max)

        # register this step's spikes (after the weight update)
        self.trace_pre  = self.trace_pre  + spikes
        self.trace_post = self.trace_post + spikes

    def apply_neuromodulation(self, modulator: "NeuromodulatorState", dt: float):
        """Three-factor consolidation (see SynapseMatrix.apply_neuromodulation)."""
        if self.w.numel() == 0:
            return
        p = self.params
        dt_ms = dt * 1000.0
        M  = p.da_gain * (modulator.dopamine - 1.0) - p.cort_gain * modulator.cortisol
        lr = p.neuromod_lr * float(np.exp(-p.cort_impair * modulator.cortisol))
        self.w = (self.w + lr * M * self.elig * dt_ms).clamp(p.w_min, p.w_max)
        if p.cort_atrophy > 0:
            self.w = self.w * (1.0 - p.cort_atrophy * modulator.cortisol * dt_ms)

    def apply_stp(self, transmit: torch.Tensor, dt: float) -> torch.Tensor:
        """Short-term plasticity per axon (identical to the dense backend)."""
        p = self.params
        dt_ms = dt * 1000.0
        self.stp_x = 1.0 - (1.0 - self.stp_x) * float(np.exp(-dt_ms / p.tau_rec))
        if p.tau_facil > 0:
            self.stp_u = p.U_stp + (self.stp_u - p.U_stp) * float(np.exp(-dt_ms / p.tau_facil))
        else:
            self.stp_u = torch.full_like(self.stp_u, p.U_stp)
        fired = transmit > 0
        self.stp_u = torch.where(fired, self.stp_u + p.U_stp * (1.0 - self.stp_u), self.stp_u)
        gain       = torch.where(fired, self.stp_u * self.stp_x, torch.zeros_like(self.stp_x))
        self.stp_x = torch.where(fired, self.stp_x - self.stp_u * self.stp_x, self.stp_x)
        return gain

    def homeostatic_scale(self, firing_rate_hz: torch.Tensor,
                          target_hz: float, eta: float):
        """Scale each neuron's incoming edges toward a target rate (Turrigiano)."""
        p = self.params
        if self.w.numel() == 0:
            return
        factor = 1.0 + eta * (target_hz - firing_rate_hz) / max(target_hz, 1e-6)
        factor = factor.clamp(0.9, 1.1)
        self.w = (self.w * factor[self.post]).clamp(p.w_min, p.w_max)

    def prune(self) -> int:
        """Remove edges whose weight decayed below threshold."""
        keep = self.w >= self.params.prune_threshold
        n_pruned = int((~keep).sum().item())
        if n_pruned:
            self.pre, self.post = self.pre[keep], self.post[keep]
            self.w, self.elig   = self.w[keep], self.elig[keep]
        return n_pruned

    # ── queries ───────────────────────────────────────────────────
    def num_synapses(self) -> int:
        return int(self.w.numel())

    def num_synapses_out(self) -> torch.Tensor:
        return torch.bincount(self.pre, minlength=self.N).float()

    def num_synapses_in(self) -> torch.Tensor:
        return torch.bincount(self.post, minlength=self.N).float()

    def weight_stats(self) -> Tuple[float, float]:
        if self.w.numel() == 0:
            return 0.0, 0.0
        return float(self.w.mean().item()), float(self.w.std().item())

    @property
    def mask(self) -> torch.Tensor:
        """Materialize the dense [N, N] existence mask (used by the growth cone).
        O(N²) memory — only call at moderate N or during periodic growth sweeps."""
        m = torch.zeros(self.N, self.N, dtype=torch.bool, device=self.device)
        if self.w.numel() > 0:
            m[self.pre, self.post] = True
        return m

    def to_dense(self) -> torch.Tensor:
        """Materialize the dense [N, N] weight matrix (for visualization)."""
        W = torch.zeros(self.N, self.N, device=self.device)
        if self.w.numel() > 0:
            W[self.pre, self.post] = self.w
        return W

    # ── persistence ───────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {'backend': 'sparse', 'pre': self.pre, 'post': self.post, 'w': self.w,
                'elig': self.elig, 'stp_u': self.stp_u, 'stp_x': self.stp_x}

    def load_state_dict(self, d: dict):
        self.pre, self.post, self.w = d['pre'], d['post'], d['w']
        self.elig = d.get('elig', torch.zeros_like(self.w))
        if 'stp_u' in d: self.stp_u = d['stp_u']
        if 'stp_x' in d: self.stp_x = d['stp_x']
