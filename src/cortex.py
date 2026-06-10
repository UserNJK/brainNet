"""
cortex.py — Structured cortical microcircuit for BrainNet
=========================================================
Turns the homogeneous random network into a model of cortex: distinct **cell
types** with their own intrinsic dynamics, wired by the canonical **E–I
microcircuit** motif, optionally organized into **columns**.

The headline payoff is biological: fast-spiking (PV⁺) interneurons providing
fast feedback inhibition to excitatory cells generate **gamma oscillations via
the PING mechanism** (pyramidal–interneuron network gamma; Cardin et al. 2009,
Buzsáki & Wang 2012) — a rhythm a homogeneous soup cannot produce.

    from cortex import build_cortex, demo_gamma
    brain = build_cortex(N=800, columns=4)
    # drive the excitatory cells and watch a gamma rhythm emerge
    early, peak = demo_gamma()

Cell types (intrinsic parameters from cortical electrophysiology):
  RS   regular-spiking pyramidal  — excitatory, slow, strongly adapting
  FS   fast-spiking PV⁺           — inhibitory, fast membrane, NO adaptation
  LTS  low-threshold-spiking SST⁺ — inhibitory, intermediate, weakly adapting
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch


@dataclass
class CellType:
    name: str
    inhibitory: bool
    fraction: float          # share of the population
    tau_m: float             # membrane time constant (ms)
    t_ref: float             # absolute refractory (ms)
    b_adapt: float           # spike-triggered adaptation increment
    V_thresh: float          # firing threshold (mV)
    description: str = ''


# Canonical cortical cell types. FS interneurons are the key new ingredient:
# fast membrane + no adaptation → high sustained rates → fast feedback inhibition.
RS  = CellType('RS',  False, 0.80, 20.0, 2.0, 0.06, -55.0,
               'regular-spiking pyramidal (excitatory)')
FS  = CellType('FS',  True,  0.15, 10.0, 1.0, 0.00, -54.0,
               'fast-spiking PV+ interneuron (fast, non-adapting)')
LTS = CellType('LTS', True,  0.05, 15.0, 2.0, 0.02, -53.0,
               'low-threshold-spiking SST+ interneuron')

DEFAULT_TYPES: List[CellType] = [RS, FS, LTS]


def _assign_types(N, cell_types, rng):
    """Return a [N] int array of cell-type indices, by fraction."""
    type_idx = np.zeros(N, dtype=int)
    order = rng.permutation(N)
    cum = 0
    for t, ct in enumerate(cell_types):
        cnt = int(round(ct.fraction * N))
        type_idx[order[cum:cum + cnt]] = t
        cum += cnt
    type_idx[order[cum:]] = 0          # leftover → first (excitatory) type
    return type_idx


def build_cortex(N: int = 800,
                 cell_types: Optional[List[CellType]] = None,
                 columns: int = 1,
                 inter_col: float = 0.15,
                 p_ee: float = 0.10, p_ei: float = 0.25,
                 p_ie: float = 0.30, p_ii: float = 0.20,
                 w_ee: float = 0.10, w_ei: float = 0.30,
                 w_ie: float = 0.50, w_ii: float = 0.30,
                 gap_degree: int = 5,
                 params=None, device=None, seed: int = 42, **brain_kwargs):
    """
    Build a BrainNet with cell-type heterogeneity and the cortical E–I
    microcircuit (probabilities/weights are per pre→post type class). `columns`
    partitions neurons into modules with dense intra- and sparse (`inter_col`×)
    inter-column connectivity. Plasticity/growth are frozen by default so the
    structure is fixed; pass params to change that. Returns the BrainNet.
    """
    try:
        from .core import BrainNet, NeuronParams
    except ImportError:
        from core import BrainNet, NeuronParams
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_types = cell_types or DEFAULT_TYPES

    if params is None:
        params = NeuronParams(
            noise_std=1.5, noise_sqrt_dt=True,
            tau_syn_i=4.5,                       # fast GABA-A → gamma-band PING
            enable_delays=True, delay_min_ms=1.0, delay_max_ms=3.0,
            seek_interval=10**9, prune_interval=10**9,   # fixed structure
            A_plus=0.0, A_minus=0.0,                     # frozen weights
        )
    brain = BrainNet(N=N, params=params, device=device, seed=seed, **brain_kwargs)
    rng = np.random.default_rng(seed)

    # ── assign cell types and their intrinsic parameters ──
    tidx = _assign_types(N, cell_types, rng)
    pick = lambda f: torch.tensor([getattr(cell_types[t], f) for t in tidx], dtype=torch.float)
    is_inh = torch.tensor([cell_types[t].inhibitory for t in tidx], dtype=torch.bool)
    brain.population.set_cell_params(
        is_inhibitory=is_inh, tau_m=pick('tau_m'), t_ref=pick('t_ref'),
        b_adapt=pick('b_adapt'), V_thresh=pick('V_thresh'))
    brain.cell_type_idx = tidx
    brain.cell_types = cell_types

    # ── structured connectivity: E–I motif × columns ──
    inh = is_inh.numpy()
    pre_inh = inh[:, None]; post_inh = inh[None, :]
    P = (p_ee * (~pre_inh & ~post_inh) + p_ei * (~pre_inh & post_inh)
         + p_ie * (pre_inh & ~post_inh) + p_ii * (pre_inh & post_inh)).astype(float)
    if columns > 1:
        col = (np.arange(N) * columns // N)
        P *= np.where(col[:, None] == col[None, :], 1.0, inter_col)
    np.fill_diagonal(P, 0.0)

    conn = rng.random((N, N)) < P
    pre_i, post_i = np.where(conn)
    pi, po = inh[pre_i], inh[post_i]
    w = np.empty(len(pre_i), dtype=np.float32)
    w[~pi & ~po] = w_ee; w[~pi & po] = w_ei
    w[pi & ~po] = w_ie;  w[pi & po] = w_ii
    w *= rng.uniform(0.7, 1.3, len(w))                 # heterogeneity

    brain.synapses.clear()
    brain.synapses.add_synapses_bulk(
        torch.tensor(pre_i), torch.tensor(post_i), torch.tensor(w))

    # ── electrical coupling (gap junctions) among FS interneurons ──
    if getattr(params, 'enable_gap', False):
        fs = np.array([i for i, t in enumerate(tidx) if cell_types[t].name == 'FS'])
        pairs = set()
        if len(fs) > 1:
            for a in fs:
                k = min(len(fs) - 1, gap_degree)
                for b in rng.choice(fs[fs != a], size=k, replace=False):
                    pairs.add((int(min(a, b)), int(max(a, b))))
            brain.population.set_gap_junctions(pairs)
        print(f"[cortex] gap junctions: {len(pairs)} FS–FS pairs")

    print(f"[cortex] {N} neurons "
          f"({int((~inh).sum())} exc / {int(inh.sum())} inh), "
          f"{brain.synapses.num_synapses()} synapses, {columns} column(s)")
    return brain


def excitatory_indices(brain) -> List[int]:
    return torch.where(~brain.population.is_inhibitory)[0].tolist()


def fast_spiking_indices(brain) -> List[int]:
    """Indices of FS (fast-spiking) interneurons, if cell types are present."""
    if not hasattr(brain, 'cell_type_idx'):
        return []
    fs = [t for t, ct in enumerate(brain.cell_types) if ct.name == 'FS']
    return [i for i, ti in enumerate(brain.cell_type_idx) if ti in fs]


# ═════════════════════════════════════════════════════════════════
#  Demo: gamma oscillations via PING
# ═════════════════════════════════════════════════════════════════

def demo_gamma(N: int = 800, seed: int = 1, device=None,
               steps: int = 4000, drive: float = 3.0, verbose: bool = True):
    """
    Drive the excitatory cells of a structured cortex and show a gamma-band
    rhythm emerge from the E–FS feedback loop (PING), with FS interneurons
    firing far faster than pyramidal cells. Returns (dominant_band, peak_hz,
    gamma_power).
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(brain, exc):
        I = torch.zeros(brain.N, device=device); I[exc] = drive
        for _ in range(steps):
            brain.step(I, dt=1e-4)
        return brain.oscillation_state()

    brain = build_cortex(N=N, columns=1, device=device, seed=seed)
    exc = excitatory_indices(brain)
    osc = run(brain, exc)

    # cell-type firing-rate check: FS should out-fire RS
    fr = brain.population.firing_rate
    fs = fast_spiking_indices(brain)
    rs = [i for i in exc]
    fs_hz = float(fr[fs].mean()) / 1e-4 if fs else 0.0
    rs_hz = float(fr[rs].mean()) / 1e-4 if rs else 0.0

    if verbose:
        bp = {k: round(v, 2) for k, v in osc['band_power'].items()}
        print(f"\n[structured cortex]  dominant={osc['dominant_band']}  "
              f"peak={osc['peak_freq_hz']:.0f} Hz  bands={bp}")
        print(f"  FS interneuron rate={fs_hz:.1f} Hz vs RS pyramidal rate={rs_hz:.1f} Hz")
    return osc['dominant_band'], osc['peak_freq_hz'], osc['band_power'].get('gamma', 0.0)


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    band, peak, gp = demo_gamma(N=n)
    print(f"\nStructured cortex rhythm: dominant={band}, peak={peak:.0f} Hz, gamma power={gp:.2f}")
