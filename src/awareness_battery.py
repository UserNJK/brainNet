"""
awareness_battery.py — Operational Markers of Minimal Awareness
===============================================================

READ THIS FIRST — what this is and is NOT
-----------------------------------------
There is **no test that proves consciousness**. The "hard problem" means no
external measurement can confirm subjective experience: any dynamical signature
could, in principle, occur with nothing being felt. This module does the
scientifically defensible thing instead — it measures the *operational markers*
that consciousness science uses to tell conscious from unconscious states in
real brains, and it validates them the only honest way: by checking that an
"awake / integrated" network **dissociates** from unconscious CONTROL states
(anaesthetised, disconnected, structure-shuffled).

A high score therefore means: "this network shows the dynamical signatures that,
in humans and animals, index the presence of consciousness." It does NOT mean
the network feels anything. Read the output as a graded, comparative
*awareness index* — only meaningful relative to the controls.

Markers (each grounded in a leading theory):
  1. Global ignition ......... Global Neuronal Workspace (Dehaene & Changeux 2011)
  2. Perturbational complexity  PCI / Lempel-Ziv (Casali et al. 2013)
  3. Information integration .. IIT-flavoured lagged mutual info  (proxy; Tononi)
  4. Predictive mismatch ...... habituation + oddball (predictive processing)
  5. Critical dynamics ........ branching ratio + spontaneous LZc (Beggs & Plenz)

Hardware: tuned for one RTX 3070 (8 GB). The dense [N,N] weight matrix is the
limit; N≈2000 is comfortable (W+age+mask ≈ 150 MB), N≈6000 still fits. Run with
device='cuda'. A small CPU demo (N≈400) runs in this file's __main__.
"""

from __future__ import annotations
import numpy as np
import torch
from core import BrainNet, NeuronParams


# ═════════════════════════════════════════════════════════════════
#  Lempel–Ziv (LZ76) complexity — the heart of the PCI / LZc markers
# ═════════════════════════════════════════════════════════════════

def lz76(binary) -> int:
    """LZ76 production count (Lempel & Ziv 1976; Kaspar & Schuster 1987)."""
    s = np.asarray(binary).astype(np.uint8).ravel()
    n = s.size
    if n < 2:
        return 1
    i = 0; k = 1; l = 1; c = 1; k_max = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0; k = 1; k_max = 1
            else:
                k = 1
    return c


def normalized_lzc(spikes_TN: np.ndarray, max_channels: int = 512,
                   rng: np.random.Generator | None = None) -> float:
    """
    Lempel-Ziv complexity of a [T, N] binary spike matrix, normalized so that a
    purely random sequence → ~1.0 and a silent/stereotyped one → ~0. Channels
    are concatenated in time (column-major), capturing temporal structure; a
    random subset is used if N is large (keeps the bit-string tractable).
    """
    arr = np.asarray(spikes_TN, dtype=bool)
    T, N = arr.shape
    if N > max_channels:
        rng = rng or np.random.default_rng(0)
        arr = arr[:, rng.choice(N, max_channels, replace=False)]
    s = arr.ravel(order='F')          # neuron-by-neuron time series, concatenated
    n = s.size
    total = int(s.sum())
    if n < 2 or total == 0 or total == n:
        return 0.0
    return float(lz76(s) * np.log2(n) / n)


# ═════════════════════════════════════════════════════════════════
#  Small helpers
# ═════════════════════════════════════════════════════════════════

def _run(brain: BrainNet, T: int, dt: float,
         stim_idx=None, stim_amp: float = 0.0, stim_window=None) -> np.ndarray:
    """Advance `brain` for T steps, optionally stimulating; return [T, N] spikes."""
    N = brain.N
    rec = np.zeros((T, N), dtype=bool)
    zero = torch.zeros(N, device=brain.device)
    stim = zero.clone()
    if stim_idx is not None:
        stim[stim_idx] = stim_amp
    for t in range(T):
        on = stim_idx is not None and (stim_window is None or stim_window[0] <= t < stim_window[1])
        spikes, _ = brain.step(stim if on else zero, dt=dt)
        rec[t] = spikes.cpu().numpy()
    return rec


def _mi(x, y, bins: int = 8) -> float:
    """Mutual information (bits) between two scalar time series via histogram."""
    cxy, _, _ = np.histogram2d(np.asarray(x, float), np.asarray(y, float), bins=bins)
    pxy = cxy / max(cxy.sum(), 1e-12)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])).sum())


def _lagged_mi(a, b, lag: int) -> float:
    return _mi(a[:-lag], b[lag:])


# ═════════════════════════════════════════════════════════════════
#  The five markers
# ═════════════════════════════════════════════════════════════════

def marker_ignition(make_brain, dt: float, amps=(0., 1.5, 3., 4.5, 6.)) -> dict:
    """
    Global-workspace ignition: graded drive to a 'sensory' subset; measure the
    LATE, post-stimulus activity in the *rest* of the network. Consciousness-like
    signature = an all-or-none jump (weak input fades locally, strong input
    ignites sustained global activity). Returns the sustain curve + a scalar
    ignition index (max sustained recruitment above the unstimulated baseline).
    """
    curve = []
    base_sustain = None
    for a in amps:
        brain = make_brain()
        N = brain.N
        sens = list(range(max(1, N // 10)))
        nonsens = np.ones(N, bool); nonsens[sens] = False
        _run(brain, 300, dt)                                  # settle
        rec = _run(brain, 600, dt, stim_idx=sens, stim_amp=a, stim_window=(100, 200))
        peak    = float(rec[100:200][:, nonsens].mean())      # during stimulus
        sustain = float(rec[260:600][:, nonsens].mean())      # AFTER stimulus offset
        if base_sustain is None:
            base_sustain = sustain
        curve.append((a, peak, sustain))
    sustains = np.array([c[2] for c in curve])
    ignition = float(sustains.max() - sustains[0])
    # nonlinearity: largest single step in the sustain curve / total range
    steps = np.diff(sustains)
    nonlin = float(steps.max() / (sustains.max() - sustains.min() + 1e-9)) if len(steps) else 0.0
    return dict(curve=curve, ignition_index=ignition, nonlinearity=nonlin)


def marker_pci(make_brain, dt: float, n_repeats: int = 4, seed: int = 0) -> dict:
    """
    Perturbational Complexity (PCI proxy, Casali et al. 2013): deliver a brief
    focal 'TMS-like' pulse and measure the spatiotemporal Lempel-Ziv complexity
    of the response. Conscious-like = high (response both spreads AND is complex);
    unconscious-like = low (no propagation, or a stereotyped global wave).
    """
    rng = np.random.default_rng(seed)
    vals = []
    for r in range(n_repeats):
        brain = make_brain()
        N = brain.N
        _run(brain, 300, dt)                                  # settle to spontaneous
        pert = list(rng.choice(N, max(1, N // 20), replace=False))
        rec = _run(brain, 300, dt, stim_idx=pert, stim_amp=10.0, stim_window=(0, 3))
        vals.append(normalized_lzc(rec[3:], rng=rng))         # exclude the pulse itself
    return dict(pci_lzc=float(np.mean(vals)), pci_std=float(np.std(vals)))


def marker_integration(make_brain, dt: float, T: int = 1500, lag: int = 5) -> dict:
    """
    Integration proxy (IIT spirit): time-lagged mutual information between the two
    halves of the network, above a temporal-shuffle baseline. High only when the
    halves genuinely co-determine each other's future (integrated whole).
    """
    brain = make_brain(); N = brain.N
    _run(brain, 300, dt)
    rec = _run(brain, T, dt)
    half = N // 2
    A = rec[:, :half].sum(1).astype(float)
    B = rec[:, half:].sum(1).astype(float)
    mi = _lagged_mi(A, B, lag) + _lagged_mi(B, A, lag)
    rng = np.random.default_rng(1)
    sh = np.mean([_lagged_mi(A, np.roll(B, int(rng.integers(lag + 1, T - lag))), lag)
                  for _ in range(5)])
    return dict(integration=float(max(0.0, mi - sh)), mi_raw=float(mi))


def marker_mismatch(make_brain, dt: float, n_standard: int = 6) -> dict:
    """
    Predictive mismatch: repeat a 'standard' stimulus (response should HABITUATE),
    then present a novel 'deviant' (response should REBOUND). A system that
    habituates and flags novelty is doing minimal predictive modelling of input.
    """
    brain = make_brain(); N = brain.N
    A = list(range(max(1, N // 10)))
    B = list(range(N // 10, N // 5))
    exclA = np.ones(N, bool); exclA[A] = False
    exclB = np.ones(N, bool); exclB[B] = False
    _run(brain, 300, dt)
    resp = []
    for _ in range(n_standard):
        rec = _run(brain, 120, dt, stim_idx=A, stim_amp=4.0, stim_window=(0, 30))
        resp.append(float(rec[30:120][:, exclA].mean()))
        _run(brain, 80, dt)                                   # inter-stimulus rest
    recd = _run(brain, 120, dt, stim_idx=B, stim_amp=4.0, stim_window=(0, 30))
    deviant = float(recd[30:120][:, exclB].mean())
    habituation = float(resp[0] - resp[-1])                   # >0 = habituated
    mismatch = float(deviant - np.mean(resp[1:]))             # >0 = novelty response
    return dict(habituation=habituation, mismatch=mismatch,
                responses=resp, deviant=deviant)


def marker_dynamics(make_brain, dt: float, T: int = 2500) -> dict:
    """
    Dynamical regime: branching ratio σ (≈1 = critical) and spontaneous Lempel-Ziv
    complexity. Awake cortex is near-critical with high spontaneous complexity;
    anaesthesia/sleep collapse both. Uses the network's built-in CriticalityAnalyzer.
    """
    brain = make_brain()
    rec = _run(brain, T, dt)
    sigma = brain.criticality_state()['branching_ratio']
    lzc = normalized_lzc(rec[-1500:])
    return dict(branching_ratio=float(sigma),
                criticality_score=float(np.exp(-abs(sigma - 1.0))),  # 1 at σ=1
                spont_lzc=float(lzc))


# ═════════════════════════════════════════════════════════════════
#  Conditions — awake/integrated vs the unconscious controls
# ═════════════════════════════════════════════════════════════════

def _awake_params(**over) -> NeuronParams:
    # Frozen plasticity + no structural growth: we probe a FIXED network's
    # dynamics, not learning. Tuned toward a lively, near-critical regime.
    p = dict(A_plus=0.0, A_minus=0.0,            # freeze STDP (no weight drift)
             seek_interval=10**9, prune_interval=10**9,   # no growth/pruning
             noise_std=1.5, noise_sqrt_dt=True,  # realistic background drive
             enable_nmda=True, nmda_ratio=0.6,   # slow recurrent drive aids ignition
             tau_nmda=120.0)
    p.update(over)
    return NeuronParams(**p)


def build_awake(N, device, seed=0, gain=2.2):
    """Intact, recurrently-coupled, near-critical network (the 'conscious' case)."""
    brain = BrainNet(N=N, params=_awake_params(), device=device, seed=seed)
    brain.synapses.W *= gain                      # set into a responsive regime
    return brain


def build_disconnected(N, device, seed=0, gain=2.2):
    """Same neurons, recurrent synapses severed → no integration or ignition."""
    brain = build_awake(N, device, seed, gain)
    brain.synapses.W.zero_()
    brain.synapses.mask.zero_()
    return brain


def build_anesthetized(N, device, seed=0, gain=2.2):
    """Hypo-excitable: strong spike-frequency adaptation + weak drive (damped,
    low-complexity responses) — an anaesthesia-like down state."""
    p = _awake_params(noise_std=0.6, b_adapt=0.35, tau_w=200.0)
    brain = BrainNet(N=N, params=p, device=device, seed=seed)
    brain.synapses.W *= gain * 0.6
    return brain


def build_shuffled(N, device, seed=0, gain=2.2):
    """Connection structure randomized (degree/weights preserved) — a non-integrated
    control with the same gross activity statistics."""
    brain = build_awake(N, device, seed, gain)
    W = brain.synapses.W
    flat = W.flatten()
    perm = torch.randperm(flat.numel(), device=W.device)
    brain.synapses.W = flat[perm].reshape(W.shape)
    brain.synapses.mask = brain.synapses.W > 0
    brain.synapses.mask.fill_diagonal_(False)
    return brain


# ═════════════════════════════════════════════════════════════════
#  Battery runner
# ═════════════════════════════════════════════════════════════════

CONDITIONS = {
    'AWAKE (integrated)': build_awake,
    'ANESTHETIZED'      : build_anesthetized,
    'DISCONNECTED'      : build_disconnected,
    'SHUFFLED'          : build_shuffled,
}

# Scalar markers extracted per condition, and whether higher = more awareness.
MARKERS = ['ignition_index', 'pci_lzc', 'integration', 'mismatch',
           'criticality_score', 'spont_lzc']


def run_battery(N=400, device=None, dt=1e-4, seed=0, verbose=True) -> dict:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"\n{'='*70}\n AWARENESS-MARKER BATTERY  (N={N}, device={device})\n{'='*70}")
        print(" Reminder: this measures correlates of consciousness and compares")
        print(" them against unconscious controls. It does NOT prove sentience.\n")

    rows = {}
    for name, builder in CONDITIONS.items():
        mk = lambda b=builder: b(N, device, seed)
        if verbose:
            print(f"  running: {name} ...")
        ign = marker_ignition(mk, dt)
        pci = marker_pci(mk, dt)
        integ = marker_integration(mk, dt)
        mis = marker_mismatch(mk, dt)
        dyn = marker_dynamics(mk, dt)
        rows[name] = dict(
            ignition_index=ign['ignition_index'],
            pci_lzc=pci['pci_lzc'],
            integration=integ['integration'],
            mismatch=mis['mismatch'],
            criticality_score=dyn['criticality_score'],
            spont_lzc=dyn['spont_lzc'],
            _detail=dict(ignition=ign, pci=pci, integration=integ,
                         mismatch=mis, dynamics=dyn),
        )

    # Aggregate: z-score each marker across conditions, average → awareness index.
    M = np.array([[rows[c][m] for m in MARKERS] for c in CONDITIONS])
    mu, sd = M.mean(0), M.std(0) + 1e-9
    Z = (M - mu) / sd
    awareness = Z.mean(1)
    for i, c in enumerate(CONDITIONS):
        rows[c]['awareness_index'] = float(awareness[i])

    if verbose:
        _print_report(rows)
    return rows


def _print_report(rows: dict):
    cols = list(rows.keys())
    print(f"\n{'-'*86}")
    print(f"{'marker':<20}" + "".join(f"{c[:16]:>17}" for c in cols))
    print(f"{'-'*86}")
    labels = {
        'ignition_index'   : 'ignition (GNW)',
        'pci_lzc'          : 'PCI / LZc',
        'integration'      : 'integration Φ*',
        'mismatch'         : 'novelty mismatch',
        'criticality_score': 'criticality σ→1',
        'spont_lzc'        : 'spontaneous LZc',
    }
    for m in MARKERS:
        print(f"{labels[m]:<20}" + "".join(f"{rows[c][m]:>17.4f}" for c in cols))
    print(f"{'-'*86}")
    print(f"{'AWARENESS INDEX':<20}" + "".join(f"{rows[c]['awareness_index']:>17.3f}" for c in cols))
    print(f"{'-'*86}")
    best = max(rows, key=lambda c: rows[c]['awareness_index'])
    awake = rows['AWAKE (integrated)']['awareness_index']
    others = max(rows[c]['awareness_index'] for c in rows if c != 'AWAKE (integrated)')
    print(f"\n  Top scorer: {best}")
    if best == 'AWAKE (integrated)' and awake > others:
        print("  ✓ The integrated network dissociates from every unconscious control")
        print("    on the aggregate index — the expected consciousness-marker pattern.")
    else:
        print("  ✗ No clean dissociation — markers do not separate awake from controls")
        print("    here. Tune `gain` / `noise_std` toward criticality and retry.")
    print("\n  Interpretation: a dissociation is EVIDENCE of consciousness-like")
    print("  DYNAMICS, not proof of experience. Use it to compare states/architectures.")


if __name__ == '__main__':
    import sys
    # Quick CPU-friendly demo by default; pass an integer N for a larger run.
    #   python awareness_battery.py            # N=400 demo
    #   python awareness_battery.py 2000        # RTX 3070-sized run (use CUDA)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    run_battery(N=N)
