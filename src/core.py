"""
BioNeuron Core — Biologically Accurate Spiking Neural Network
=============================================================
Each neuron in this network operates like a real cortical neuron:

  Dendrites     → receive weighted synaptic currents
  Soma (LIF)    → leaky integrate-and-fire membrane potential
  Adaptation    → spike-frequency adaptation (AHP / M-current, K+)
  Axon Hillock  → fires action potential when threshold is crossed
  Axon          → optional finite conduction delay to targets
  Growth Cone   → actively seeks neurons with correlated spike trains
  Receptors     → fast AMPA + slow voltage-gated NMDA (optional)
  Short-term    → Tsodyks–Markram facilitation/depression (optional)
  Synapse       → STDP weight updates (Hebbian pre→post) — not backprop
  Pruning       → weak unused synapses die
  Homeostasis   → synaptic scaling holds firing near a set-point
  Reward/stress → dopamine potentiates / cortisol depresses eligible synapses
  Modulation    → dopamine/serotonin/norepinephrine/cortisol alter global state

Population dynamics:
  Oscillations  → LFP-proxy spectrum (delta…gamma) for rhythm/binding analysis
  Criticality   → branching ratio + neuronal avalanches (Beggs & Plenz)
  Statistics    → ISI irregularity (CV) and Fano factor (firing regime)
  Backend       → dense [N,N] or sparse edge-list (scales past the N² ceiling)

Thought Formation (Cell Assemblies, Hebb 1949):
  Neurons that fire together, wire together.
  Assemblies = groups of synchronously active neurons.
  These are detected in real-time as "thoughts".

Companion modules (compose with this one):
  sparse_synapses.py → edge-list backend to scale past the dense [N,N] ceiling
  neurochemistry.py  → extensible neurochemical system (drop-in for the modulator)
  closed_loop.py     → learning from self-generated feedback (closed reward loop)
  cortex.py          → cell types (FS interneurons) + cortical microcircuit (gamma)

Single-neuron model: leaky integrate-and-fire, optionally Adaptive Exponential
(AdEx) with spike-frequency adaptation, NMDA, gap junctions, two-compartment
active dendrites, and per-neuron cell-type heterogeneity. Plasticity: STDP,
inhibitory STDP, short-term plasticity, homeostatic scaling, and three-factor
reward/stress learning.

Author  : Jugal Kishore
Version : 1.8.0
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings

# ─────────────────────────────────────────────
#  BIOLOGICAL CONSTANTS (based on cortical data)
# ─────────────────────────────────────────────

@dataclass
class NeuronParams:
    """
    Biophysical parameters for a cortical neuron.
    All time constants in milliseconds, potentials in mV.
    """
    # Membrane
    V_rest     : float = -70.0   # Resting potential (mV)
    V_thresh   : float = -55.0   # Firing threshold (mV)
    V_reset    : float = -75.0   # Post-spike reset (mV)
    V_peak     : float =  30.0   # Action potential peak (mV)
    tau_m      : float =  20.0   # Membrane time constant (ms)
    C_m        : float =   1.0   # Membrane capacitance (normalized)

    # Refractory
    t_ref      : float =   2.0   # Absolute refractory period (ms)

    # Synaptic current decay
    tau_syn_e  : float =   5.0   # Excitatory synapse time constant (ms)
    tau_syn_i  : float =  10.0   # Inhibitory synapse time constant (ms)

    # STDP (Spike-Timing Dependent Plasticity)
    A_plus     : float =  0.01   # LTP amplitude
    A_minus    : float =  0.012  # LTD amplitude (slightly stronger = stability)
    tau_plus   : float =  20.0   # LTP time window (ms)
    tau_minus  : float =  20.0   # LTD time window (ms)
    w_max      : float =   1.0   # Max synaptic weight
    w_min      : float =   0.0   # Min synaptic weight

    # Growth cone (axon seeking)
    seek_corr_threshold  : float = 0.35   # Min spike-train correlation to form synapse
    seek_interval        : int   = 50     # Steps between growth cone sweeps
    max_synapses_out     : int   = 50     # Max outgoing synapses per neuron
    max_synapses_in      : int   = 100    # Max incoming synapses per neuron (dendritic tree)

    # Pruning
    prune_threshold      : float = 0.005  # Weights below this → synapse dies
    prune_interval       : int   = 200    # Steps between pruning passes

    # Noise (thermal / background synaptic noise)
    noise_std            : float = 0.5    # Injected current noise (pA)
    # Proper stochastic (Euler–Maruyama) scaling: the noise term should grow as
    # √dt, not dt. With this ON, `noise_std` becomes a true diffusion coefficient
    # and modest values (~1–2) drive realistic spontaneous activity. OFF keeps
    # the original dt-scaled behavior (so prior experiments reproduce exactly).
    noise_sqrt_dt        : bool  = False

    # Dale's Law ratio
    inhibitory_fraction  : float = 0.20   # 20% of neurons are inhibitory (GABAergic)

    # ── Spike-frequency adaptation (AHP / M-current, K+-mediated) ──
    # Real cortical neurons fire fast then slow down — a Ca²⁺/Na⁺-activated K⁺
    # current hyperpolarises the cell after each spike (Brette & Gerstner 2005,
    # adaptation variable `w`). This is the single most ubiquitous deviation
    # from pure LIF, so it is ON by default.
    enable_adaptation    : bool  = True
    a_adapt              : float =  0.0    # subthreshold adaptation coupling (a*(V-V_rest))
    b_adapt              : float =  0.04   # spike-triggered adaptation increment
    tau_w                : float = 120.0   # adaptation current time constant (ms)

    # ── Exponential spike initiation (Adaptive Exponential I&F; Brette & Gerstner 2005) ──
    # Adds the sharp voltage-dependent upstroke term (Δ_T/τ_m)·exp((V−V_T)/Δ_T),
    # promoting the adaptive-LIF to the standard AdEx model. Combined with the
    # adaptation parameters it reproduces tonic / adapting / bursting / initial-
    # burst firing (Naud et al. 2008). Spikes are detected at V_peak. Opt-in.
    enable_exp           : bool  = False
    delta_T              : float =   2.0   # spike slope factor (mV)
    V_T                  : float = -55.0   # soft threshold for the exponential (mV)

    # ── Conductance-based synapses (reversal potentials) ──
    # If True : I_syn = g_e*(E_exc - V) + g_i*(E_inh - V)  → realistic, gives
    #           shunting (divisive) inhibition and voltage-dependent drive.
    # If False: current-based I_syn = g_e - g_i  (default — preserves the input
    #           scaling the existing notebook/experiments are tuned for).
    conductance_based    : bool  = False
    E_exc                : float =   0.0   # excitatory reversal potential (mV, AMPA)
    E_inh                : float = -80.0   # inhibitory reversal potential (mV, GABA_A)

    # ── Axonal conduction delays (per-presynaptic-neuron) ──
    # Spikes take finite time to travel down the axon (≈0.5–20 ms in cortex).
    # Delays are essential for travelling waves and gamma/theta rhythms.
    enable_delays        : bool  = False
    delay_min_ms         : float =  1.0
    delay_max_ms         : float =  5.0

    # ── Homeostatic synaptic scaling (Turrigiano & Nelson 2004) ──
    # Slow multiplicative rescaling of a neuron's *incoming* weights to keep its
    # firing rate near a set-point. Counteracts STDP's tendency to run away to
    # silence or saturation. Off by default (slow, alters long-run dynamics).
    enable_homeostasis   : bool  = False
    target_rate_hz       : float =  5.0    # desired mean firing rate (Hz)
    homeo_interval       : int   = 500     # steps between scaling passes
    homeo_eta            : float =  0.01   # scaling rate (fraction per pass)

    # ── Short-term synaptic plasticity (Tsodyks–Markram; Mongillo et al. 2008) ──
    # Per-axon facilitation (u) and depression (x): transmitted efficacy = u·x.
    # Defaults are a *facilitating* synapse (the working-memory regime); set
    # tau_facil → 0 for a depression-dominated synapse. Opt-in.
    enable_stp           : bool  = False
    U_stp                : float =   0.2   # baseline utilisation / facilitation step
    tau_rec              : float = 200.0   # depression (resource) recovery (ms)
    tau_facil            : float = 1500.0  # facilitation decay (ms); 0 = no facilitation

    # ── NMDA receptors (slow, voltage-gated excitatory conductance) ──
    # A second excitatory channel with Mg²⁺ block (Jahr & Stevens 1990) on top
    # of the fast AMPA-like conductance. Always uses the reversal-potential form
    # (NMDA is intrinsically voltage-dependent). Enables coincidence detection
    # and the slow recurrent drive that sustains attractor states. Opt-in.
    enable_nmda          : bool  = False
    tau_nmda             : float = 100.0   # NMDA conductance decay (ms)
    nmda_ratio           : float =   0.5   # NMDA strength relative to AMPA
    Mg                   : float =   1.0   # extracellular [Mg²⁺] (mM)

    # ── Inhibitory plasticity (iSTDP; Vogels et al. 2011) ──
    # A symmetric plasticity rule on inhibitory→postsynaptic synapses that grows
    # inhibition onto over-active cells and shrinks it onto quiet ones, driving
    # each postsynaptic neuron toward `rho_target` Hz — detailed E/I balance.
    # Stabilises STDP/reward learning (a fix for runaway excitation). Opt-in.
    enable_istdp         : bool  = False
    eta_istdp            : float =  0.001  # inhibitory learning rate
    tau_istdp            : float = 20.0    # iSTDP trace time constant (ms)
    rho_target           : float =  5.0    # target postsynaptic firing rate (Hz)

    # ── Dendritic computation (two-compartment soma + active dendrite) ──
    # Real pyramidal neurons are not points: excitation arrives on dendrites that
    # integrate it nonlinearly and fire regenerative NMDA/Ca²⁺ "dendritic spikes"
    # (plateau potentials), which then drive the soma — making a single neuron a
    # two-layer computation (Poirazi & Mel 2003; Larkum 2013). Here excitation
    # (incl. NMDA) charges a separate dendritic compartment coupled to the soma;
    # inhibition stays perisomatic (PV⁺ basket cells). Opt-in.
    enable_dendrite      : bool  = False
    tau_d                : float = 25.0    # dendritic membrane time constant (ms)
    g_couple             : float =  1.0    # soma↔dendrite coupling conductance
    V_d_thresh           : float = -40.0   # dendritic-spike threshold (mV)
    dend_plateau         : float =  3.0    # plateau (dendritic-spike) drive
    tau_dend_spike       : float = 40.0    # plateau / dendritic-spike duration (ms)

    # ── Gap junctions (electrical synapses between interneurons) ──
    # Bidirectional ohmic coupling I_gap = g_gap·Σ(V_j − V_i) over coupled pairs;
    # synchronises fast-spiking interneurons and sharpens gamma (Galarreta &
    # Hestrin 1999). Wired by cortex.py among FS cells. Opt-in.
    enable_gap           : bool  = False
    g_gap                : float =  0.02   # gap-junction conductance (per coupled pair)

    # ── Connectivity backend ──
    # False: dense [N,N] weight matrix (simple, fine to a few thousand neurons).
    # True : sparse edge-list backend (O(E) memory & per-step cost) — needed to
    #        scale past the dense [N,N] ceiling on an 8 GB GPU. Same dynamics.
    sparse_synapses      : bool  = False

    # ── Reward / stress neuromodulated learning (three-factor; Frémaux & Gerstner 2016) ──
    # When ON, STDP no longer edits weights directly — it lays down a decaying
    # *eligibility trace* tagging recently co-active synapses. A later dopamine
    # (reward) or cortisol (stress) signal converts those tags into potentiation
    # or depression, so credit lands on the synapses that drove the outcome.
    # Drive it with brain.reward() / brain.punish() / brain.feedback(correct).
    # Opt-in (default behaviour = direct dopamine-scaled STDP, unchanged).
    enable_reward_learning : bool  = False
    tau_elig             : float = 1000.0  # eligibility-trace decay (ms) = credit window
    neuromod_lr          : float =    0.002 # consolidation rate (Δw per ms per tag·signal)
    da_gain              : float =    1.0   # dopamine → potentiation gain
    cort_gain            : float =    1.0   # cortisol → depression gain
    tau_dopamine         : float =  200.0   # phasic dopamine relaxes to baseline (ms)
    tau_cortisol         : float = 3000.0   # cortisol is a slow hormone — long decay (ms)
    cort_impair          : float =    0.3   # high cortisol impairs plasticity: lr·exp(−cort·this)
    cort_atrophy         : float =    0.0   # chronic-cortisol synaptic atrophy (per ms); 0 = off


@dataclass
class NeuromodulatorState:
    """
    Global neuromodulator levels (1.0 = baseline for the multiplicative ones).

    dopamine        : reward signal → potentiates eligible synapses (LTP)
    serotonin       : mood/calm     → scales tau_m (slower = more stable)
    norepinephrine  : arousal       → scales noise and firing threshold
    acetylcholine   : attention     → scales seek_corr_threshold (more synapse-forming)
    cortisol        : stress/punish → SLOW glucocorticoid (HPA axis). Depresses
                      recently-active (eligible) synapses (LTD) and, when
                      sustained, impairs plasticity. Baseline 0.0 (no stress),
                      rises on error/punishment. The negative counterpart to
                      dopamine in reward-modulated learning.
    """
    dopamine        : float = 1.0
    serotonin       : float = 1.0
    norepinephrine  : float = 1.0
    acetylcholine   : float = 1.0
    cortisol        : float = 0.0


# ─────────────────────────────────────────────
#  SYNAPSE
# ─────────────────────────────────────────────

class SynapseMatrix:
    """
    Sparse dynamic synapse population for N neurons.

    Uses a dense weight matrix but a sparse mask so only
    existing synapses are computed. Synapse counts stay
    biologically bounded (max_out, max_in per neuron).

    Stores:
        W       : [N, N] weight matrix (float32)
        mask    : [N, N] bool — which synapses exist
        trace_pre  : [N]  pre-synaptic eligibility trace
        trace_post : [N]  post-synaptic eligibility trace
    """

    def __init__(self, N: int, params: NeuronParams, device: torch.device):
        self.N      = N
        self.params = params
        self.device = device

        # Weight matrix — initially empty
        self.W    = torch.zeros(N, N, device=device)
        self.mask = torch.zeros(N, N, dtype=torch.bool, device=device)

        # No self-connections
        self.mask.fill_diagonal_(False)

        # STDP eligibility traces
        self.trace_pre  = torch.zeros(N, device=device)
        self.trace_post = torch.zeros(N, device=device)

        # Synapse age tracking (for pruning decisions)
        self.age = torch.zeros(N, N, device=device)

        # Short-term plasticity state, per presynaptic axon (Tsodyks–Markram)
        self.stp_u = torch.full((N,), params.U_stp, device=device)  # utilisation
        self.stp_x = torch.ones(N, device=device)                   # resources

        # Eligibility trace for three-factor (reward/stress) learning — lazily
        # built only when enable_reward_learning is on (it is another [N,N]).
        self.elig = None

        # Inhibitory-STDP trace, per neuron (Vogels et al. 2011)
        self.istdp_trace = torch.zeros(N, device=device)

    def add_synapse(self, pre: int, post: int, weight: Optional[float] = None):
        """Form a new synapse from neuron pre → post."""
        if pre == post:
            return
        if weight is None:
            # Small random initial weight, biologically realistic
            weight = float(torch.empty(1).uniform_(0.01, 0.1))
        self.W[pre, post]    = weight
        self.mask[pre, post] = True
        self.age[pre, post]  = 0.0

    def add_synapses_bulk(self, pre: torch.Tensor, post: torch.Tensor,
                          weights: Optional[torch.Tensor] = None):
        """Vectorized creation of many synapses at once (drops self-loops)."""
        pre  = pre.to(self.device); post = post.to(self.device)
        keep = pre != post
        pre, post = pre[keep], post[keep]
        if weights is None:
            weights = torch.empty(pre.shape, device=self.device).uniform_(0.01, 0.1)
        else:
            weights = weights.to(self.device)[keep]
        self.W[pre, post]    = weights
        self.mask[pre, post] = True
        self.age[pre, post]  = 0.0

    def clear(self):
        """Remove all synapses (e.g. to build structured connectivity from scratch)."""
        self.W.zero_(); self.mask.zero_(); self.age.zero_()
        if self.elig is not None:
            self.elig.zero_()

    def remove_synapse(self, pre: int, post: int):
        self.W[pre, post]    = 0.0
        self.mask[pre, post] = False
        self.age[pre, post]  = 0.0

    def num_synapses_out(self) -> torch.Tensor:
        return self.mask.float().sum(dim=1)   # [N]

    def num_synapses_in(self) -> torch.Tensor:
        return self.mask.float().sum(dim=0)   # [N]

    # ── Backend-agnostic interface (shared with SparseSynapses) ──
    def propagate(self, spike_f: torch.Tensor) -> torch.Tensor:
        """Postsynaptic input I_post[j] = Σ_i spike_f[i] · W[i,j]."""
        return torch.matmul(spike_f, self.W * self.mask.float())

    def num_synapses(self) -> int:
        return int(self.mask.sum().item())

    def weight_stats(self) -> Tuple[float, float]:
        """(mean, std) of existing synaptic weights."""
        if not bool(self.mask.any()):
            return 0.0, 0.0
        w = self.W[self.mask]
        return float(w.mean().item()), float(w.std().item())

    def state_dict(self) -> dict:
        return {'backend': 'dense', 'W': self.W, 'mask': self.mask, 'age': self.age,
                'stp_u': self.stp_u, 'stp_x': self.stp_x}

    def load_state_dict(self, d: dict):
        self.W, self.mask = d['W'], d['mask']
        if 'age'   in d: self.age   = d['age']
        if 'stp_u' in d: self.stp_u = d['stp_u']
        if 'stp_x' in d: self.stp_x = d['stp_x']

    def apply_stdp(self, spikes: torch.Tensor, dt: float,
                   modulator: NeuromodulatorState):
        """
        Spike-Timing Dependent Plasticity (Bi & Poo 1998; online pair rule,
        Morrison et al. 2008).

        W is indexed [pre, post]. Each neuron k keeps two traces:
            trace_pre[k]  : presynaptic eligibility  (decays with tau_plus)
            trace_post[k] : postsynaptic eligibility (decays with tau_minus)

        Causality (the whole point of STDP):
          - post fires now & pre fired *before*  (Δt = t_post − t_pre > 0)
                → LTP:  Δw[pre, post] += A_plus  * trace_pre[pre]
          - pre fires now  & post fired *before*  (Δt < 0)
                → LTD:  Δw[pre, post] −= A_minus * trace_post[post]

        Traces are updated with the current spikes *after* the weight change is
        computed, so a synapse is driven by past partner activity, not by the
        coincident spike itself (correct nearest-/all-to-all pair ordering).

        Dopamine scales LTP amplitude (reward-modulated STDP, Schultz 1997).
        """
        p = self.params
        dt_ms = dt * 1000.0  # convert to ms

        # ── Decay eligibility traces ──
        self.trace_pre  = self.trace_pre  * float(np.exp(-dt_ms / p.tau_plus))
        self.trace_post = self.trace_post * float(np.exp(-dt_ms / p.tau_minus))

        # ── Weight change from THIS step's spikes against the *prior* traces ──
        # LTP: post[j] fires now, weighted by how recently pre[i] fired.
        ltp = (self.trace_pre.unsqueeze(1) * spikes.unsqueeze(0) * p.A_plus)
        # LTD: pre[i] fires now, weighted by how recently post[j] fired.
        ltd = (spikes.unsqueeze(1) * self.trace_post.unsqueeze(0) * p.A_minus)

        if p.enable_reward_learning:
            # Three-factor: tag synapses with recent *causal* (pre→post)
            # coactivation in a slow eligibility trace; dopamine (reward) or
            # cortisol (stress) then converts the tag into potentiation or
            # depression later, in apply_neuromodulation. Using the LTP-shaped
            # (causal) component keeps the tag positive for the active pathway,
            # so reward strengthens it and cortisol weakens it.
            if self.elig is None:
                self.elig = torch.zeros_like(self.W)
            self.elig = (self.elig * float(np.exp(-dt_ms / p.tau_elig))
                         + ltp * self.mask.float())
        else:
            # Direct reward-modulated STDP: dopamine scales potentiation.
            dW = (ltp * modulator.dopamine - ltd) * self.mask.float()
            self.W = (self.W + dW).clamp(p.w_min, p.w_max)
            self.W = self.W * self.mask.float()   # zero out non-synapses

        # ── Register this step's spikes into the traces (after the update) ──
        self.trace_pre  = self.trace_pre  + spikes
        self.trace_post = self.trace_post + spikes

        # Age existing synapses
        self.age = self.age + self.mask.float() * dt_ms

    def apply_neuromodulation(self, modulator: NeuromodulatorState, dt: float):
        """
        Three-factor consolidation (Frémaux & Gerstner 2016): convert the
        eligibility trace into weight change, gated by the global reward/stress
        signal  M = da_gain·(dopamine−1) − cort_gain·cortisol.

            reward  (dopamine > 1) → M > 0 → eligible synapses POTENTIATE (LTP)
            punish  (cortisol > 0) → M < 0 → eligible synapses DEPRESS  (LTD)

        Sustained cortisol additionally impairs plasticity (lower effective rate)
        and can cause slow synaptic atrophy — modelling chronic-stress effects.
        """
        if self.elig is None:
            return
        p = self.params
        dt_ms = dt * 1000.0
        M  = p.da_gain * (modulator.dopamine - 1.0) - p.cort_gain * modulator.cortisol
        lr = p.neuromod_lr * float(np.exp(-p.cort_impair * modulator.cortisol))
        self.W = (self.W + lr * M * self.elig * dt_ms).clamp(p.w_min, p.w_max)
        if p.cort_atrophy > 0:
            self.W = self.W * (1.0 - p.cort_atrophy * modulator.cortisol * dt_ms)
        self.W = self.W * self.mask.float()

    def apply_istdp(self, spikes: torch.Tensor, dt: float, is_inh: torch.Tensor):
        """
        Inhibitory STDP (Vogels et al. 2011). A symmetric rule on inhibitory→post
        synapses that drives each postsynaptic neuron toward `rho_target`: when a
        target is over-active its inhibition is strengthened, when quiet it is
        weakened (a baseline depression −α), establishing detailed E/I balance.
        is_inh : [N] bool — which presynaptic neurons are inhibitory.
        """
        p = self.params
        dt_ms = dt * 1000.0
        self.istdp_trace = self.istdp_trace * float(np.exp(-dt_ms / p.tau_istdp))
        alpha = 2.0 * (p.rho_target / 1000.0) * p.tau_istdp     # depression offset
        inh = is_inh.float()
        # presynaptic (inhibitory) spike: Δw[i,j] += η·(trace[j] − α)
        t1 = (spikes * inh).unsqueeze(1) * (self.istdp_trace.unsqueeze(0) - alpha)
        # postsynaptic spike: Δw[i,j] += η·trace[i]   (inhibitory i only)
        t2 = (inh * self.istdp_trace).unsqueeze(1) * spikes.unsqueeze(0)
        self.W = (self.W + p.eta_istdp * (t1 + t2) * self.mask.float()).clamp(p.w_min, p.w_max)
        self.W = self.W * self.mask.float()
        self.istdp_trace = self.istdp_trace + spikes

    def homeostatic_scale(self, firing_rate_hz: torch.Tensor,
                          target_hz: float, eta: float):
        """
        Homeostatic synaptic scaling (Turrigiano & Nelson 2004).

        Multiplicatively rescales each neuron's *incoming* synapses (a column of
        W) to nudge its firing rate toward `target_hz`. Unlike STDP this is
        non-competitive — it preserves relative weights while regulating the
        overall excitability set-point, preventing STDP from collapsing the
        network into silence or saturation.

        firing_rate_hz : [N] current mean firing rate per neuron (Hz)
        """
        p = self.params
        # Under-active neurons scale incoming weights up, over-active scale down.
        factor = 1.0 + eta * (target_hz - firing_rate_hz) / max(target_hz, 1e-6)
        factor = factor.clamp(0.9, 1.1)                 # gentle, bounded step
        self.W = self.W * factor.unsqueeze(0)           # column j = inputs to j
        self.W = (self.W * self.mask.float()).clamp(p.w_min, p.w_max)

    def apply_stp(self, transmit: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Short-term plasticity (Tsodyks–Markram; Mongillo et al. 2008), per axon.

        On each transmitted spike the utilisation u facilitates (u → u+U(1−u))
        and the resource pool x is depleted (x → x − u·x); both relax back
        between spikes (x toward 1 with tau_rec, u toward U with tau_facil). The
        per-spike release efficacy is u·x, so trains depress, facilitate, or
        both depending on the time constants.

        transmit : [N] float (0/1) — spikes transmitted this step
        Returns  : [N] release gain (u·x for firing axons, 0 otherwise)
        """
        p = self.params
        dt_ms = dt * 1000.0

        # Recover toward baselines (all axons)
        self.stp_x = 1.0 - (1.0 - self.stp_x) * float(np.exp(-dt_ms / p.tau_rec))
        if p.tau_facil > 0:
            self.stp_u = p.U_stp + (self.stp_u - p.U_stp) * float(np.exp(-dt_ms / p.tau_facil))
        else:
            self.stp_u = torch.full_like(self.stp_u, p.U_stp)

        fired  = transmit > 0
        # Facilitation jump, then release = u(post-jump)·x(pre-depletion)
        self.stp_u = torch.where(fired, self.stp_u + p.U_stp * (1.0 - self.stp_u), self.stp_u)
        gain       = torch.where(fired, self.stp_u * self.stp_x, torch.zeros_like(self.stp_x))
        self.stp_x = torch.where(fired, self.stp_x - self.stp_u * self.stp_x, self.stp_x)
        return gain

    def prune(self):
        """Remove synapses that have decayed below threshold."""
        dead = (self.W < self.params.prune_threshold) & self.mask
        num_pruned = dead.sum().item()
        self.mask[dead] = False
        self.W[dead]    = 0.0
        return int(num_pruned)


# ─────────────────────────────────────────────
#  POPULATION — Leaky Integrate and Fire
# ─────────────────────────────────────────────

class NeuronPopulation:
    """
    Vectorized population of N Leaky Integrate-and-Fire neurons.

    Membrane dynamics (Euler integration):
        dV/dt = (V_rest - V) / tau_m  +  I_total / C_m

    I_total = I_syn (from spikes) + I_ext (external input) + I_noise

    Fires when V >= V_thresh.
    Refractory: neuron is locked at V_reset for t_ref ms after firing.

    Dale's Law enforced: ~20% inhibitory (GABA), rest excitatory (Glu).
    """

    def __init__(self, N: int, params: NeuronParams,
                 modulator: NeuromodulatorState, device: torch.device):
        self.N         = N
        self.params    = params
        self.modulator = modulator
        self.device    = device

        # Membrane potential — start at rest with small noise
        self.V = torch.full((N,), params.V_rest, device=device)
        self.V += torch.randn(N, device=device) * 2.0

        # Refractory counter (ms remaining)
        self.ref_count = torch.zeros(N, device=device)

        # Synaptic conductances
        self.g_e    = torch.zeros(N, device=device)   # fast excitatory (AMPA)
        self.g_i    = torch.zeros(N, device=device)   # inhibitory (GABA)
        self.g_nmda = torch.zeros(N, device=device)   # slow excitatory (NMDA)

        # Spike-frequency adaptation current (AHP / M-current, K+)
        self.w_adapt = torch.zeros(N, device=device)

        # Dendritic compartment (two-compartment model): voltage + plateau timer
        self.V_d = torch.full((N,), params.V_rest, device=device)
        self.dend_timer = torch.zeros(N, device=device)   # ms left of a dendritic spike

        # Axonal conduction-delay state (lazily built on first step — needs dt)
        self._delay_buf   = None   # [L, N] ring buffer of emitted spikes
        self._delay_steps = None   # [N]    per-neuron axonal delay, in steps
        self._delay_wptr  = 0

        # Per-neuron parameter overrides for cell-type heterogeneity (None → use
        # the scalar NeuronParams value). Set via set_cell_params() / cortex.py.
        self.tau_m_vec    = None   # [N] membrane time constant (ms)
        self.t_ref_vec    = None   # [N] absolute refractory period (ms)
        self.b_adapt_vec  = None   # [N] spike-triggered adaptation increment
        self.V_thresh_vec = None   # [N] firing threshold (mV)

        # Gap-junction (electrical synapse) directed pair list (None = none).
        self.gap_pre  = None       # [G] long
        self.gap_post = None       # [G] long

        # Dale's Law: assign neuron types
        n_inh = int(N * params.inhibitory_fraction)
        inh_idx = torch.randperm(N)[:n_inh]
        self.is_inhibitory = torch.zeros(N, dtype=torch.bool, device=device)
        self.is_inhibitory[inh_idx] = True

        # Sign vector: excitatory = +1, inhibitory = -1
        self.sign = torch.where(self.is_inhibitory,
                                torch.tensor(-1.0, device=device),
                                torch.tensor( 1.0, device=device))

        # Spike history for correlation (spike trains, ms resolution)
        self.spike_history = torch.zeros(N, 500, dtype=torch.bool, device=device)
        self.history_ptr   = 0

        # Firing rate tracker (exponential moving average)
        self.firing_rate = torch.zeros(N, device=device)

    def set_cell_params(self, is_inhibitory=None, tau_m=None, t_ref=None,
                        b_adapt=None, V_thresh=None):
        """
        Assign per-neuron (cell-type) parameters. Each argument is a [N] tensor,
        or None to leave that parameter at the scalar NeuronParams default. Used
        by cortex.py to give fast-spiking interneurons their distinct, fast,
        non-adapting dynamics, etc.
        """
        if is_inhibitory is not None:
            self.is_inhibitory = is_inhibitory.to(self.device).bool()
            self.sign = torch.where(self.is_inhibitory,
                                    torch.tensor(-1.0, device=self.device),
                                    torch.tensor( 1.0, device=self.device))
        if tau_m    is not None: self.tau_m_vec    = tau_m.to(self.device).float()
        if t_ref    is not None: self.t_ref_vec    = t_ref.to(self.device).float()
        if b_adapt  is not None: self.b_adapt_vec  = b_adapt.to(self.device).float()
        if V_thresh is not None: self.V_thresh_vec = V_thresh.to(self.device).float()

    def set_gap_junctions(self, pairs):
        """
        Wire electrical synapses. `pairs` is a list/array of undirected (a, b)
        index pairs; each is stored in both directions so coupling is symmetric.
        """
        pairs = np.asarray(list(pairs), dtype=np.int64).reshape(-1, 2)
        a, b = pairs[:, 0], pairs[:, 1]
        pre  = np.concatenate([a, b]); post = np.concatenate([b, a])
        self.gap_pre  = torch.tensor(pre,  device=self.device)
        self.gap_post = torch.tensor(post, device=self.device)

    def step(self, I_ext: torch.Tensor, synapses: SynapseMatrix,
             dt: float) -> torch.Tensor:
        """
        Advance simulation by dt seconds.

        Args:
            I_ext  : [N] external input current
            synapses: SynapseMatrix with current weights
            dt     : timestep in seconds

        Returns:
            spikes : [N] bool tensor — which neurons fired this step
        """
        p      = self.params
        mod    = self.modulator
        dt_ms  = dt * 1000.0

        # ── Neurons currently in refractory cannot integrate ──
        in_ref = self.ref_count > 0
        self.ref_count = (self.ref_count - dt_ms).clamp(min=0.0)

        # ── Synaptic conductance decay ──
        self.g_e = self.g_e * float(np.exp(-dt_ms / p.tau_syn_e))
        self.g_i = self.g_i * float(np.exp(-dt_ms / p.tau_syn_i))
        if p.enable_nmda:
            self.g_nmda = self.g_nmda * float(np.exp(-dt_ms / p.tau_nmda))

        # Per-neuron (cell-type) parameter overrides, falling back to scalars.
        tau_m_   = self.tau_m_vec    if self.tau_m_vec    is not None else p.tau_m
        b_adapt_ = self.b_adapt_vec  if self.b_adapt_vec  is not None else p.b_adapt
        V_thr_   = self.V_thresh_vec if self.V_thresh_vec is not None else p.V_thresh
        t_ref_   = (self.t_ref_vec if self.t_ref_vec is not None
                    else torch.tensor(p.t_ref, device=self.device))

        # ── Membrane integration ──
        # Modulate tau_m by serotonin (calm = slower integration)
        tau_m_eff = tau_m_ * mod.serotonin

        # Leak current
        I_leak = (p.V_rest - self.V) / tau_m_eff

        if p.enable_dendrite:
            # ── Two-compartment: excitation (+NMDA) → dendrite, inhibition → soma ──
            # Excitatory drive integrated on the dendritic compartment.
            I_exc_d = (self.g_e * (p.E_exc - self.V_d)) if p.conductance_based else self.g_e
            if p.enable_nmda:
                B_d = 1.0 / (1.0 + p.Mg * torch.exp(-0.062 * self.V_d) / 3.57)
                I_exc_d = I_exc_d + p.nmda_ratio * self.g_nmda * B_d * (p.E_exc - self.V_d)
            dend_active = self.dend_timer > 0
            I_plateau = torch.where(dend_active,
                                    torch.full_like(self.V_d, p.dend_plateau),
                                    torch.zeros_like(self.V_d))
            I_leak_d = (p.V_rest - self.V_d) / p.tau_d
            dV_d = (I_leak_d + I_exc_d / p.C_m
                    + p.g_couple * (self.V - self.V_d) + I_plateau) * dt_ms
            self.V_d = (self.V_d + dV_d).clamp(max=p.V_peak)
            # Regenerative dendritic spike: V_d crossing threshold opens a plateau.
            trig = (self.V_d >= p.V_d_thresh) & (~dend_active)
            self.dend_timer = torch.where(
                trig, torch.full_like(self.dend_timer, p.tau_dend_spike),
                (self.dend_timer - dt_ms).clamp(min=0.0))
            # Soma sees perisomatic inhibition + coupling from the dendrite.
            I_syn = (-self.g_i / p.C_m) + p.g_couple * (self.V_d - self.V)
        else:
            # Single-compartment: conductance- or current-based net synaptic input.
            if p.conductance_based:
                I_syn = (self.g_e * (p.E_exc - self.V) +
                         self.g_i * (p.E_inh - self.V)) / p.C_m
            else:
                I_syn = (self.g_e - self.g_i) / p.C_m
            # Slow NMDA current: voltage-gated by Mg²⁺ block (Jahr & Stevens 1990).
            if p.enable_nmda:
                B = 1.0 / (1.0 + p.Mg * torch.exp(-0.062 * self.V) / 3.57)
                I_syn = I_syn + p.nmda_ratio * self.g_nmda * B * (p.E_exc - self.V) / p.C_m

        # Spike-frequency adaptation: a hyperpolarising K+ current that grows
        # with each spike and decays with tau_w (Brette & Gerstner 2005).
        I_adapt = (self.w_adapt / p.C_m) if p.enable_adaptation else 0.0

        # AdEx exponential spike-initiation term: a sharp, voltage-dependent
        # upstroke that activates near V_T (Brette & Gerstner 2005).
        if p.enable_exp:
            I_exp = (p.delta_T / tau_m_eff) * torch.exp(
                ((self.V - p.V_T) / p.delta_T).clamp(max=20.0))
        else:
            I_exp = 0.0

        # Gap junctions: diffusive electrical coupling I_gap = g_gap·Σ(V_j − V_i).
        if p.enable_gap and self.gap_pre is not None and self.gap_pre.numel() > 0:
            diff = self.V[self.gap_post] - self.V[self.gap_pre]
            I_gap = torch.zeros(self.N, device=self.device).index_add_(
                0, self.gap_pre, p.g_gap * diff)
        else:
            I_gap = 0.0

        # Deterministic drift, integrated over dt
        dV_det = (I_leak + I_exp + I_syn - I_adapt + I_gap + I_ext) * dt_ms

        # Thermal / background noise (norepinephrine scales arousal noise).
        # Stochastic term scales as √dt (Euler–Maruyama) when enabled, else dt.
        noise_scale = np.sqrt(dt_ms) if p.noise_sqrt_dt else dt_ms
        dV_noise = (torch.randn(self.N, device=self.device) *
                    p.noise_std * mod.norepinephrine * noise_scale)

        # Total integration
        dV = dV_det + dV_noise
        V_new = self.V + dV
        if p.enable_exp:
            V_new = V_new.clamp(max=p.V_peak)        # cap the exponential runaway
        self.V = torch.where(in_ref, torch.tensor(p.V_reset, device=self.device), V_new)

        # ── Spike detection ──
        if p.enable_exp:
            # AdEx: the exponential drives V to V_peak; detect the spike there.
            spikes = (self.V >= p.V_peak) & (~in_ref)
        else:
            # Threshold modulated by norepinephrine (arousal lowers threshold)
            V_thresh_eff = V_thr_ - (mod.norepinephrine - 1.0) * 3.0
            spikes = (self.V >= V_thresh_eff) & (~in_ref)

        # ── Post-spike reset ──
        self.V        = torch.where(spikes, torch.tensor(p.V_reset, device=self.device), self.V)
        self.ref_count = torch.where(spikes, t_ref_, self.ref_count)

        # ── Adaptation update: relax toward a*(V−V_rest), kick by b on a spike ──
        if p.enable_adaptation:
            dw = (p.a_adapt * (self.V - p.V_rest) - self.w_adapt) / p.tau_w
            self.w_adapt = self.w_adapt + dw * dt_ms + spikes.float() * b_adapt_

        # ── Propagate spikes via synapses ──
        # Each firing neuron injects current scaled by its sign (Dale's Law)
        # into all its postsynaptic targets. With delays enabled, what arrives
        # this step is the spikes emitted one axonal-delay ago.
        arriving = self._apply_delay(spikes, dt_ms) if p.enable_delays else spikes.float()
        # Short-term plasticity scales each axon's release by its current u·x.
        if p.enable_stp:
            spike_f = self.sign * synapses.apply_stp(arriving, dt)  # [N], signed
        else:
            spike_f = arriving * self.sign                          # [N], signed
        # I_post[j] = sum_i W[i,j] * spike_f[i]   (dense matmul or sparse scatter)
        I_post   = synapses.propagate(spike_f)  # [N]

        # Separate into excitatory and inhibitory
        exc_in = I_post.clamp(min=0)
        inh_in = (-I_post).clamp(min=0)
        self.g_e = self.g_e + exc_in
        self.g_i = self.g_i + inh_in
        if p.enable_nmda:                              # same drive feeds slow NMDA
            self.g_nmda = self.g_nmda + exc_in

        # ── Update spike history ──
        self.spike_history[:, self.history_ptr] = spikes
        self.history_ptr = (self.history_ptr + 1) % self.spike_history.size(1)

        # ── Update firing rate EMA ──
        self.firing_rate = 0.99 * self.firing_rate + 0.01 * spikes.float()

        return spikes

    def _apply_delay(self, spikes: torch.Tensor, dt_ms: float) -> torch.Tensor:
        """
        Per-axon conduction delay. Each presynaptic neuron is assigned one fixed
        delay (drawn once from [delay_min_ms, delay_max_ms]); its spikes reach
        all downstream targets that many steps later. Returns the spike vector
        *arriving at terminals this step*. The ring buffer is built lazily on
        first call because the delay-in-steps depends on dt.
        """
        p = self.params
        if self._delay_buf is None:
            d_steps = torch.round(
                torch.empty(self.N, device=self.device).uniform_(
                    p.delay_min_ms, p.delay_max_ms) / dt_ms
            ).long().clamp(min=1)
            self._delay_steps = d_steps
            L = int(d_steps.max().item()) + 1
            self._delay_buf  = torch.zeros(L, self.N, device=self.device)
            self._delay_wptr = 0

        L = self._delay_buf.size(0)
        self._delay_buf[self._delay_wptr] = spikes.float()           # emit now
        read_idx = (self._delay_wptr - self._delay_steps) % L        # [N]
        arriving = self._delay_buf[read_idx,
                                   torch.arange(self.N, device=self.device)]
        self._delay_wptr = (self._delay_wptr + 1) % L
        return arriving


# ─────────────────────────────────────────────
#  GROWTH CONE — Axon Seeking
# ─────────────────────────────────────────────

class GrowthCone:
    """
    Simulates axon growth cone dynamics.

    In biology:
        - Growth cones navigate toward chemical gradients
        - Activity-dependent: they home in on neurons with correlated signals
        - Once close enough (correlation > threshold), a synapse forms

    Here:
        - "Signal" = recent spike train (binary vector over time window)
        - "Gradient" = spike-train cross-correlation
        - If corr(i, j) > theta AND no synapse exists AND capacity not exceeded
          → form synapse

    Acetylcholine lowers threshold (more synapse-seeking = more attention/plasticity)
    """

    def __init__(self, params: NeuronParams, modulator: NeuromodulatorState,
                 device: torch.device):
        self.params    = params
        self.modulator = modulator
        self.device    = device
        self.step_count = 0
        self.synapses_formed  = 0
        self.synapses_pruned  = 0

    def sweep(self, population: NeuronPopulation, synapses: SynapseMatrix):
        """
        Full growth cone sweep across the population.
        Computes pairwise spike-train correlations and forms/prunes synapses.
        """
        self.step_count += 1
        p   = self.params
        mod = self.modulator
        N   = population.N

        # ── Get recent spike history ──
        # Use last 200 timesteps for correlation window
        history = population.spike_history.float()   # [N, T]

        # Normalize (z-score each neuron's train)
        mu  = history.mean(dim=1, keepdim=True)
        std = history.std(dim=1, keepdim=True).clamp(min=1e-6)
        h_norm = (history - mu) / std                # [N, T]

        # ── Pairwise correlation matrix ──
        # corr[i,j] = how correlated are neurons i and j
        T    = history.size(1)
        corr = torch.matmul(h_norm, h_norm.t()) / T  # [N, N]
        corr.fill_diagonal_(0.0)                      # no self-connections

        # Acetylcholine lowers the threshold (more seeking)
        threshold = p.seek_corr_threshold / mod.acetylcholine

        # ── Form new synapses where correlation exceeds threshold ──
        n_out = synapses.num_synapses_out()   # [N]
        n_in  = synapses.num_synapses_in()    # [N]

        candidates = (
            (corr > threshold) &
            (~synapses.mask) &                           # no synapse yet
            (~torch.eye(N, dtype=torch.bool, device=self.device))  # no self
        )

        # Respect capacity limits
        over_out = (n_out >= p.max_synapses_out).unsqueeze(1).expand_as(candidates)
        over_in  = (n_in  >= p.max_synapses_in ).unsqueeze(0).expand_as(candidates)
        candidates = candidates & (~over_out) & (~over_in)

        new_pre, new_post = candidates.nonzero(as_tuple=True)

        # Initial weight proportional to correlation strength
        for pre, post in zip(new_pre.tolist(), new_post.tolist()):
            w_init = float(corr[pre, post].item()) * 0.1
            synapses.add_synapse(pre, post, weight=w_init)
            self.synapses_formed += 1

        # ── Prune weak synapses ──
        prune_every = max(1, p.prune_interval // p.seek_interval)
        if self.step_count % prune_every == 0:
            pruned = synapses.prune()
            self.synapses_pruned += pruned

        return len(new_pre)


# ─────────────────────────────────────────────
#  THOUGHT DETECTOR — Cell Assembly Detection
# ─────────────────────────────────────────────

class ThoughtDetector:
    """
    Detects 'thoughts' as Cell Assemblies (Hebb, 1949).

    A cell assembly is a group of neurons that:
      1. Fire synchronously (correlated spike times)
      2. Are recurrently connected to each other

    Algorithm:
      - Build co-firing matrix: C[i,j] = how often i and j spike together
      - Threshold to get binary co-firing graph
      - Find connected components = assemblies (thoughts)
      - Each assembly is a "thought" — label it by dominant frequency pattern

    Stability of an assembly over time = thought persistence (working memory analog).
    """

    def __init__(self, N: int, window: int = 50, threshold: float = 0.3,
                 device: torch.device = torch.device('cpu')):
        self.N         = N
        self.window    = window
        self.threshold = threshold
        self.device    = device
        self.assemblies     : List[List[int]] = []
        self.assembly_history: deque = deque(maxlen=200)

    def detect(self, spike_history: torch.Tensor) -> List[List[int]]:
        """
        spike_history: [N, T] bool

        Returns list of assemblies (each = list of neuron indices).
        """
        # Use last `window` timesteps
        recent = spike_history[:, -self.window:].float()   # [N, window]

        # Normalize
        mu  = recent.mean(dim=1, keepdim=True)
        std = recent.std(dim=1, keepdim=True).clamp(min=1e-6)
        r_norm = (recent - mu) / std

        # Co-firing correlation
        C = torch.matmul(r_norm, r_norm.t()) / self.window  # [N, N]
        C.fill_diagonal_(0)

        # Binary graph
        G = (C > self.threshold).cpu().tolist()

        # Connected components via BFS
        visited    = [False] * self.N
        assemblies = []

        for start in range(self.N):
            if visited[start] or not any(G[start]):
                continue
            component = []
            queue     = [start]
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                neighbors = [i for i, val in enumerate(G[node]) if val]
                queue.extend([n for n in neighbors if not visited[n]])

            if len(component) >= 3:   # min assembly size
                assemblies.append(sorted(component))

        self.assemblies = assemblies
        self.assembly_history.append(len(assemblies))
        return assemblies

    def assembly_stability(self) -> float:
        """
        How stable are the thought assemblies over time?
        Low variance = stable (working memory), high variance = chaotic.
        """
        if len(self.assembly_history) < 10:
            return 0.0
        counts = list(self.assembly_history)
        arr    = np.array(counts, dtype=float)
        if arr.std() < 1e-6:
            return 1.0
        # Coefficient of variation, inverted (lower CV = more stable)
        cv = arr.std() / (arr.mean() + 1e-6)
        return float(np.exp(-cv))


# ─────────────────────────────────────────────
#  OSCILLATION ANALYZER — Population Rhythms (LFP proxy)
# ─────────────────────────────────────────────

class OscillationAnalyzer:
    """
    Detects population rhythms — the network analogue of an EEG/LFP spectrum.

    Cortical field potentials are dominated by summed synaptic / population
    activity, and that signal carries the classic frequency bands:

        delta 0.5–4 Hz | theta 4–8 | alpha 8–13 | beta 13–30 | gamma 30–100

    The LFP proxy used here is the population firing rate (spike count per
    timestep) — the standard multi-unit signal whose power spectrum exposes
    network oscillations and gamma/theta "binding" (Buzsáki & Draguhn 2004).

    Note on resolution: frequency resolution is 1 / (window · dt). With the
    default window=2048 and dt=1e-4 s that is ≈ 4.9 Hz — fine for beta/gamma,
    coarse for delta/theta. Widen `window` for better low-frequency detail.
    """

    BANDS = {
        'delta': (0.5,   4.0),
        'theta': (4.0,   8.0),
        'alpha': (8.0,  13.0),
        'beta' : (13.0, 30.0),
        'gamma': (30.0, 100.0),
    }

    F_LO = 0.5    # low edge of the neural analysis range (Hz)

    def __init__(self, window: int = 2048, f_max: float = 100.0):
        self.window = window
        self.f_max  = f_max                        # high edge of neural range (Hz)
        self.signal: deque = deque(maxlen=window)
        self.dt: Optional[float] = None

    def push(self, population_rate: float, dt: float):
        """Record one LFP-proxy sample (population spike count this step)."""
        self.signal.append(float(population_rate))
        self.dt = dt

    def analyze(self) -> dict:
        """
        Spectral summary, restricted to the neural range [F_LO, f_max] so that
        band powers are comparable regardless of the (much higher) sampling
        rate. Returns:

            band_power      : relative power per band, summing to ~1 *within*
                              the neural range (i.e. of the rhythmic activity,
                              how is it split across delta…gamma)
            dominant_band   : band carrying the most in-range power
            peak_freq_hz    : strongest frequency in the neural range
            rhythmicity     : fraction of in-range power in the single peak bin
                              (high = a sharp, dominant rhythm)
            inband_fraction : in-range power / total power — how oscillatory the
                              population is overall vs broadband/asynchronous
        """
        empty = {
            'band_power'     : {b: 0.0 for b in self.BANDS},
            'dominant_band'  : None,
            'peak_freq_hz'   : 0.0,
            'rhythmicity'    : 0.0,
            'inband_fraction': 0.0,
        }
        n = len(self.signal)
        if n < 64 or self.dt is None:
            return empty

        x = np.asarray(self.signal, dtype=float)
        x = x - x.mean()                           # remove DC offset
        if np.allclose(x, 0.0):
            return empty
        x = x * np.hanning(n)                       # taper to limit spectral leakage

        psd   = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(n, d=self.dt)

        total_all = float(psd[1:].sum()) + 1e-12    # all freqs, excluding DC
        in_range  = (freqs >= self.F_LO) & (freqs <= self.f_max)
        idx       = np.where(in_range)[0]
        if idx.size == 0:                           # too few samples to resolve
            return empty                            # any neural-range frequency yet
        inband    = float(psd[in_range].sum()) + 1e-12

        # Bands tile the neural range exactly; the top band is inclusive up to
        # f_max so every in-range bin lands in exactly one band (powers sum ~1).
        band_power = {}
        items = list(self.BANDS.items())
        for k, (band, (lo, hi)) in enumerate(items):
            if k == len(items) - 1:
                sel = (freqs >= lo) & (freqs <= self.f_max)
            else:
                sel = (freqs >= lo) & (freqs < hi)
            band_power[band] = float(psd[sel].sum() / inband)

        peak_bin = int(idx[np.argmax(psd[idx])])
        return {
            'band_power'     : band_power,
            'dominant_band'  : max(band_power, key=band_power.get),
            'peak_freq_hz'   : float(freqs[peak_bin]),
            'rhythmicity'    : float(psd[peak_bin] / inband),
            'inband_fraction': float(inband / total_all),
        }


# ─────────────────────────────────────────────
#  CRITICALITY ANALYZER — Self-Organized Criticality
# ─────────────────────────────────────────────

class CriticalityAnalyzer:
    """
    Tests whether the network operates near a critical point (Beggs & Plenz
    2003). Healthy cortex sits at the boundary between dying-out and runaway
    activity, where:

        branching ratio σ ≈ 1   (each spike triggers ~1 descendant on average)
        avalanche sizes follow a power law  P(s) ~ s^(-1.5)

    σ < 1 = subcritical (activity fades) ; σ > 1 = supercritical (seizure-like).
    Fed the population spike count each step; computes on demand.
    """

    def __init__(self, window: int = 4096):
        self.counts: deque = deque(maxlen=window)

    def push(self, n_spikes: float):
        self.counts.append(int(n_spikes))

    def analyze(self) -> dict:
        out = {'branching_ratio': 0.0, 'n_avalanches': 0,
               'mean_avalanche': 0.0, 'max_avalanche': 0, 'powerlaw_slope': None}
        n = np.asarray(self.counts, dtype=float)
        if n.size < 32:
            return out

        # Branching ratio as the least-squares slope of n(t+1) vs n(t).
        a, b = n[:-1], n[1:]
        denom = float((a * a).sum())
        if denom > 1e-9:
            out['branching_ratio'] = float((a * b).sum() / denom)

        # Avalanches = runs of consecutive active bins, separated by silence.
        sizes, cur = [], 0
        for c in self.counts:
            if c > 0:
                cur += c
            elif cur > 0:
                sizes.append(cur); cur = 0
        if cur > 0:
            sizes.append(cur)

        if sizes:
            s = np.asarray(sizes, dtype=float)
            out.update(n_avalanches=len(sizes),
                       mean_avalanche=float(s.mean()),
                       max_avalanche=int(s.max()))
            if len(sizes) >= 20 and s.max() > 1:
                # Coarse power-law exponent: slope of the log-log size histogram.
                edges = np.logspace(0, np.log10(s.max() + 1), 12)
                hist, _ = np.histogram(s, bins=edges)
                centers = np.sqrt(edges[:-1] * edges[1:])
                m = hist > 0
                if m.sum() >= 3:
                    out['powerlaw_slope'] = float(
                        np.polyfit(np.log10(centers[m]), np.log10(hist[m]), 1)[0])
        return out


# ─────────────────────────────────────────────
#  SPIKE STATISTICS — Firing Regime Diagnostics
# ─────────────────────────────────────────────

class SpikeStatistics:
    """
    Spike-train statistics that reveal the network's dynamical regime.

    Cortex in vivo fires irregularly and asynchronously (Softky & Koch 1993;
    Brunel 2000's asynchronous-irregular state):

        CV(ISI) ≈ 1   (Poisson-like; ≪1 = clock-like, >1 = bursty)
        Fano    ≈ 1   (spike-count variance ≈ mean)

    Computed from a chronologically-ordered [N, T] spike history.
    """

    @staticmethod
    def compute(spike_history: np.ndarray, dt: float) -> dict:
        sh = spike_history.astype(bool)
        N, T = sh.shape
        total = int(sh.sum())
        rate_hz = float(total / (N * T * dt)) if (N * T) > 0 else 0.0

        # CV of inter-spike intervals, averaged over neurons with ≥3 spikes.
        cvs = []
        for i in range(N):
            t = np.flatnonzero(sh[i])
            if t.size >= 3:
                isi = np.diff(t).astype(float)
                if isi.mean() > 0:
                    cvs.append(isi.std() / isi.mean())
        cv_isi = float(np.mean(cvs)) if cvs else 0.0

        # Fano factor: per-neuron spike-count variance / mean across time bins.
        fano = 0.0
        nbins = 10
        if T >= nbins:
            usable = T - (T % nbins)
            binned = sh[:, :usable].reshape(N, nbins, -1).sum(axis=2).astype(float)
            m = binned.mean(axis=1); v = binned.var(axis=1)
            active = m > 0
            if active.any():
                fano = float(np.mean(v[active] / m[active]))

        return {
            'mean_rate_hz': rate_hz,
            'cv_isi'      : cv_isi,
            'fano_factor' : fano,
            'n_active'    : int((sh.sum(axis=1) > 0).sum()),
        }


# ─────────────────────────────────────────────
#  MAIN BRAIN — Full Orchestration
# ─────────────────────────────────────────────

class BrainNet:
    """
    Full biologically accurate spiking neural network.

    Usage:
        brain = BrainNet(N=500)
        for t in range(T):
            I_ext = your_input_signal(t)         # [N] tensor
            spikes, state = brain.step(I_ext)
            print(state)                          # consciousness metrics

    Key properties:
        brain.modulator.dopamine = 1.5           # reward signal
        brain.modulator.acetylcholine = 1.8      # attention / seeking mode
        brain.get_thoughts()                     # current cell assemblies
        brain.get_state()                        # full metrics dict
    """

    def __init__(self, N: int = 500,
                 params: Optional[NeuronParams] = None,
                 device: Optional[torch.device] = None,
                 seed: int = 42,
                 neurochem: Optional[object] = None):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.N      = N
        self.params = params or NeuronParams()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t      = 0        # simulation time (ms)
        self.step_i = 0        # step counter

        print(f"[BrainNet] Initializing {N} neurons on {self.device}")

        # Neuromodulator state. Default is the simple 5-scalar NeuromodulatorState;
        # pass a Neurochemistry (neurochemistry.py) for the full chemical system —
        # it is API-compatible (exposes the same axis attributes) and advances its
        # own kinetics each step.
        self.modulator = neurochem if neurochem is not None else NeuromodulatorState()

        # Core components — dense or sparse connectivity backend
        if self.params.sparse_synapses:
            try:
                from .sparse_synapses import SparseSynapses   # installed package
            except ImportError:
                from sparse_synapses import SparseSynapses    # flat / script use
            Synapses = SparseSynapses
        else:
            Synapses = SynapseMatrix
        self.population  = NeuronPopulation(N, self.params, self.modulator, self.device)
        self.synapses    = Synapses(N, self.params, self.device)
        self.growth      = GrowthCone(self.params, self.modulator, self.device)
        self.thoughts    = ThoughtDetector(N, device=self.device)
        self.oscillation = OscillationAnalyzer()
        self.criticality = CriticalityAnalyzer()
        self._last_osc   = self.oscillation.analyze()
        self._last_crit  = self.criticality.analyze()

        # ── Seed initial sparse random connectivity (like newborn brain) ──
        # ~5% connection probability, random weights (vectorized bulk insert)
        init_prob = 0.05
        n_init    = int(N * N * init_prob)
        pre_idx   = torch.randint(0, N, (n_init,))
        post_idx  = torch.randint(0, N, (n_init,))
        self.synapses.add_synapses_bulk(pre_idx, post_idx)

        print(f"[BrainNet] Initial synapses: {self.synapses.num_synapses()}")
        print(f"[BrainNet] Inhibitory neurons: {self.population.is_inhibitory.sum().item()}")
        print("[BrainNet] Ready.\n")

        # Metrics log
        self._log: Dict[str, List] = {
            'time'           : [],
            'n_spikes'       : [],
            'n_synapses'     : [],
            'n_thoughts'     : [],
            'thought_size'   : [],
            'stability'      : [],
            'firing_rate'    : [],
            'peak_freq_hz'   : [],
            'gamma_power'    : [],
            'mean_adaptation': [],
            'branching_ratio': [],
        }

    # ──────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────

    def step(self, I_ext: Optional[torch.Tensor] = None,
             dt: float = 1e-4) -> Tuple[torch.Tensor, dict]:
        """
        Advance simulation by one timestep.

        Args:
            I_ext : [N] input current (default: zero)
            dt    : timestep in seconds (default 0.1 ms)

        Returns:
            spikes : [N] bool — neurons that fired
            state  : dict of current metrics
        """
        if I_ext is None:
            I_ext = torch.zeros(self.N, device=self.device)
        else:
            I_ext = I_ext.to(self.device)

        self._last_dt = dt

        # ── LIF integration + spike detection ──
        spikes   = self.population.step(I_ext, self.synapses, dt)
        n_spikes = int(spikes.sum().item())

        # ── STDP weight updates ──
        self.synapses.apply_stdp(spikes.float(), dt, self.modulator)

        # ── Inhibitory plasticity (detailed E/I balance) ──
        if self.params.enable_istdp:
            self.synapses.apply_istdp(spikes.float(), dt, self.population.is_inhibitory)

        # ── Three-factor consolidation: dopamine/cortisol gate eligibility ──
        if self.params.enable_reward_learning:
            self.synapses.apply_neuromodulation(self.modulator, dt)

        # ── Neuromodulator dynamics ──
        # A Neurochemistry advances its own kinetics (all chemicals clear toward
        # baseline). A plain NeuromodulatorState gets the legacy phasic decay,
        # but only while reward learning is active (otherwise levels are held).
        if hasattr(self.modulator, 'step'):
            self.modulator.step(dt)
        elif self.params.enable_reward_learning:
            dt_ms = dt * 1000.0
            self.modulator.dopamine = 1.0 + (self.modulator.dopamine - 1.0) * \
                float(np.exp(-dt_ms / self.params.tau_dopamine))
            self.modulator.cortisol = self.modulator.cortisol * \
                float(np.exp(-dt_ms / self.params.tau_cortisol))

        # ── Homeostatic synaptic scaling (slow, periodic) ──
        if (self.params.enable_homeostasis and self.step_i > 0 and
                self.step_i % self.params.homeo_interval == 0):
            self.synapses.homeostatic_scale(self.population.firing_rate / dt,
                                            self.params.target_rate_hz,
                                            self.params.homeo_eta)

        # ── Feed the LFP / oscillation + criticality analyzers ──
        self.oscillation.push(n_spikes, dt)
        self.criticality.push(n_spikes)

        # ── Growth cone sweep (periodic) ──
        new_synapses = 0
        if self.step_i % self.params.seek_interval == 0:
            new_synapses = self.growth.sweep(self.population, self.synapses)

        # ── Thought + oscillation detection (periodic) ──
        assemblies = []
        if self.step_i % 25 == 0:
            assemblies      = self.thoughts.detect(self.population.spike_history)
            self._last_osc  = self.oscillation.analyze()
            self._last_crit = self.criticality.analyze()

        mean_adapt = (float(self.population.w_adapt.mean().item())
                      if self.params.enable_adaptation else 0.0)

        # ── Log ──
        if self.step_i % 10 == 0:
            self._log['time'].append(self.t)
            self._log['n_spikes'].append(n_spikes)
            self._log['n_synapses'].append(self.synapses.num_synapses())
            self._log['n_thoughts'].append(len(assemblies))
            self._log['thought_size'].append(
                np.mean([len(a) for a in assemblies]) if assemblies else 0
            )
            self._log['stability'].append(self.thoughts.assembly_stability())
            self._log['firing_rate'].append(self.population.firing_rate.mean().item())
            self._log['peak_freq_hz'].append(self._last_osc['peak_freq_hz'])
            self._log['gamma_power'].append(self._last_osc['band_power']['gamma'])
            self._log['mean_adaptation'].append(mean_adapt)
            self._log['branching_ratio'].append(self._last_crit['branching_ratio'])

        self.t      += dt * 1000.0   # advance time in ms
        self.step_i += 1

        state = {
            'time_ms'          : self.t,
            'n_spikes'         : n_spikes,
            'n_synapses'       : self.synapses.num_synapses(),
            'n_thoughts'       : len(assemblies),
            'assemblies'       : assemblies,
            'thought_stability': self.thoughts.assembly_stability(),
            'new_synapses'     : new_synapses,
            'mean_firing_hz'   : float(self.population.firing_rate.mean().item()) / dt,
            'dominant_rhythm'  : self._last_osc['dominant_band'],
            'peak_freq_hz'     : self._last_osc['peak_freq_hz'],
            'mean_adaptation'  : mean_adapt,
            'branching_ratio'  : self._last_crit['branching_ratio'],
        }

        return spikes, state

    # ──────────────────────────────────────────
    #  EXTERNAL INTERFACES
    # ──────────────────────────────────────────

    def stimulate(self, neuron_indices: List[int], amplitude: float = 5.0,
                  dt: float = 1e-4) -> Tuple[torch.Tensor, dict]:
        """
        Stimulate specific neurons (like sensory input or electrode stimulation).
        """
        I = torch.zeros(self.N, device=self.device)
        I[neuron_indices] = amplitude
        return self.step(I, dt)

    def reward(self, value: float = 1.5, decay: float = 0.05):
        """
        Release dopamine — a reward signal. With reward-learning ON, this
        POTENTIATES the synapses tagged eligible by recent activity (three-factor
        LTP), so the pathway that just fired gets reinforced. With it OFF (default),
        dopamine directly scales STDP's LTP, as before.
        value : dopamine level (1.0 = baseline, >1 = reward).
        """
        self.modulator.dopamine = value

    def punish(self, value: float = 1.0):
        """
        Release cortisol — a stress/punishment signal (slow glucocorticoid). With
        reward-learning ON, it DEPRESSES the synapses tagged eligible by recent
        activity (three-factor LTD), weakening the pathway that just fired, and
        when sustained it impairs further plasticity. The negative counterpart to
        reward(). `value` is the stress level (0 = none); it accumulates and
        decays slowly (tau_cortisol).
        """
        self.modulator.cortisol = max(self.modulator.cortisol, value)

    def feedback(self, correct: bool, magnitude: float = 1.0,
                 use_cortisol: bool = False):
        """
        Teaching signal applied to whichever synapses the network's recent
        activity made *eligible*. Requires enable_reward_learning=True.

        correct → dopamine burst (potentiate the active pathway).
        wrong   → by default a dopamine *dip* below baseline — a fast, phasic
                  reward-prediction-error (Schultz 1997) that drives LTD and is
                  what makes trial-by-trial learning work. Pass use_cortisol=True
                  to instead raise the slow stress hormone (models stress; note
                  it accumulates and can cancel later reward — chronic-stress
                  impairment of learning).

            for trial in trials:
                brain.stimulate(stimulus); ... ; brain.feedback(was_correct)
        """
        if correct:
            self.reward(1.0 + magnitude)
        elif use_cortisol:
            self.punish(magnitude)
        else:
            self.modulator.dopamine = max(0.0, 1.0 - magnitude)   # phasic RPE dip → LTD

    def attend(self, level: float = 1.5):
        """
        Release acetylcholine — enter attention mode.
        Lowers synapse-formation threshold (growth cones seek more aggressively).
        """
        self.modulator.acetylcholine = level

    def calm(self, level: float = 1.3):
        """Serotonin release — slower integration, more stability."""
        self.modulator.serotonin = level

    def arouse(self, level: float = 1.4):
        """Norepinephrine — lower threshold, more noise = creative mode."""
        self.modulator.norepinephrine = level

    def get_thoughts(self) -> List[List[int]]:
        """Return current cell assemblies (thoughts)."""
        return self.thoughts.detect(self.population.spike_history)

    def get_dominant_thought(self) -> Optional[List[int]]:
        """Return the largest current cell assembly."""
        assemblies = self.get_thoughts()
        if not assemblies:
            return None
        return max(assemblies, key=len)

    def oscillation_state(self) -> dict:
        """
        Current population-rhythm spectrum: relative power per band
        (delta/theta/alpha/beta/gamma), dominant band, and spectral peak (Hz).
        """
        return self.oscillation.analyze()

    def criticality_state(self) -> dict:
        """
        Self-organized-criticality diagnostics: branching ratio (σ≈1 = critical)
        and neuronal-avalanche statistics (Beggs & Plenz 2003).
        """
        return self.criticality.analyze()

    def spike_statistics(self) -> dict:
        """
        Firing-regime diagnostics over the spike-history window: mean rate,
        ISI irregularity CV (≈1 = Poisson-like), and Fano factor.
        """
        ptr = self.population.history_ptr
        sh  = self.population.spike_history
        # Roll so the oldest sample is first (chronological order for ISIs).
        chrono = torch.cat([sh[:, ptr:], sh[:, :ptr]], dim=1).cpu().numpy()
        return SpikeStatistics.compute(chrono, getattr(self, '_last_dt', 1e-4))

    def get_state(self) -> dict:
        """
        Snapshot of current network metrics *without* advancing the simulation.
        (The same metrics are also returned live from each step().)
        """
        dt  = getattr(self, '_last_dt', 1e-4)
        osc = self.oscillation.analyze()
        return {
            'time_ms'          : self.t,
            'step'             : self.step_i,
            'n_synapses'       : self.synapses.num_synapses(),
            'n_thoughts'       : len(self.thoughts.assemblies),
            'thought_stability': self.thoughts.assembly_stability(),
            'mean_firing_hz'   : float(self.population.firing_rate.mean().item()) / dt,
            'dominant_rhythm'  : osc['dominant_band'],
            'peak_freq_hz'     : osc['peak_freq_hz'],
            'band_power'       : osc['band_power'],
            'criticality'      : self.criticality.analyze(),
            'spike_stats'      : self.spike_statistics(),
            'connectivity'     : self.connectivity_stats(),
        }

    def connectivity_stats(self) -> dict:
        n_syn          = self.synapses.num_synapses()
        w_mean, w_std  = self.synapses.weight_stats()
        max_out = int(self.synapses.num_synapses_out().max().item())
        max_in  = int(self.synapses.num_synapses_in().max().item())
        return dict(
            n_synapses=n_syn, w_mean=w_mean, w_std=w_std,
            max_out=max_out, max_in=max_in,
            density=n_syn / (self.N * (self.N - 1)),
            formed_total=self.growth.synapses_formed,
            pruned_total=self.growth.synapses_pruned,
        )

    def get_log(self) -> dict:
        return {k: np.array(v) for k, v in self._log.items()}

    def save(self, path: str):
        """Save full network state (works for both dense and sparse backends)."""
        torch.save({
            'V'             : self.population.V,
            'V_d'           : self.population.V_d,
            'w_adapt'       : self.population.w_adapt,
            'g_nmda'        : self.population.g_nmda,
            'firing_rate'   : self.population.firing_rate,
            'synapses'      : self.synapses.state_dict(),
            'modulator'     : self.modulator.__dict__,
            't'             : self.t,
            'step_i'        : self.step_i,
            'params'        : self.params.__dict__,
        }, path)
        print(f"[BrainNet] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.population.V = ckpt['V']
        if 'V_d'         in ckpt: self.population.V_d         = ckpt['V_d']
        if 'w_adapt'     in ckpt: self.population.w_adapt     = ckpt['w_adapt']
        if 'g_nmda'      in ckpt: self.population.g_nmda      = ckpt['g_nmda']
        if 'firing_rate' in ckpt: self.population.firing_rate = ckpt['firing_rate']
        if 'synapses' in ckpt:                       # current format
            self.synapses.load_state_dict(ckpt['synapses'])
        elif 'W' in ckpt:                            # legacy dense checkpoint
            self.synapses.load_state_dict(ckpt)
        self.t      = ckpt['t']
        self.step_i = ckpt['step_i']
        print(f"[BrainNet] Loaded from {path}")
