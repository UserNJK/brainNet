"""
BioNeuron Core — Biologically Accurate Spiking Neural Network
=============================================================
Each neuron in this network operates like a real cortical neuron:

  Dendrites     → receive weighted synaptic currents
  Soma (LIF)    → leaky integrate-and-fire membrane potential
  Axon Hillock  → fires action potential when threshold is crossed
  Growth Cone   → actively seeks neurons with correlated spike trains
  Synapse       → STDP-based weight updates (not backprop)
  Pruning       → weak unused synapses die
  Modulation    → dopamine/serotonin/norepinephrine alter global state

Thought Formation (Cell Assemblies, Hebb 1949):
  Neurons that fire together, wire together.
  Assemblies = groups of synchronously active neurons.
  These are detected in real-time as "thoughts".

Author  : Jugal Kishore
Version : 1.0.0
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

    # Dale's Law ratio
    inhibitory_fraction  : float = 0.20   # 20% of neurons are inhibitory (GABAergic)


@dataclass
class NeuromodulatorState:
    """
    Global neuromodulator levels (0.0 → 2.0 range, 1.0 = baseline).

    dopamine        : reward signal → scales LTP (A_plus)
    serotonin       : mood/calm     → scales tau_m (slower = more stable)
    norepinephrine  : arousal       → scales noise and firing threshold
    acetylcholine   : attention     → scales seek_corr_threshold (more synapse-forming)
    """
    dopamine        : float = 1.0
    serotonin       : float = 1.0
    norepinephrine  : float = 1.0
    acetylcholine   : float = 1.0


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

    def remove_synapse(self, pre: int, post: int):
        self.W[pre, post]    = 0.0
        self.mask[pre, post] = False
        self.age[pre, post]  = 0.0

    def num_synapses_out(self) -> torch.Tensor:
        return self.mask.float().sum(dim=1)   # [N]

    def num_synapses_in(self) -> torch.Tensor:
        return self.mask.float().sum(dim=0)   # [N]

    def apply_stdp(self, spikes: torch.Tensor, dt: float,
                   modulator: NeuromodulatorState):
        """
        Spike-Timing Dependent Plasticity.

        For every synapse (pre, post):
          - if pre just fired: Δw += A_plus * trace_post[post]  (LTP)
          - if post just fired: Δw -= A_minus * trace_pre[pre]  (LTD)

        Eligibility traces decay exponentially between spikes.
        Dopamine scales LTP amplitude (reward-modulated STDP).
        """
        p = self.params
        dt_ms = dt * 1000.0  # convert to ms

        # Decay traces
        self.trace_pre  = self.trace_pre  * torch.exp(torch.tensor(-dt_ms / p.tau_plus,  device=self.device))
        self.trace_post = self.trace_post * torch.exp(torch.tensor(-dt_ms / p.tau_minus, device=self.device))

        # Add spike contribution
        self.trace_pre  = self.trace_pre  + spikes
        self.trace_post = self.trace_post + spikes

        # LTP: pre fired → strengthen all its outgoing synapses
        # dW[pre, post] += A_plus * trace_post[post]
        ltp = (spikes.unsqueeze(1) *
               self.trace_post.unsqueeze(0) *
               p.A_plus * modulator.dopamine)

        # LTD: post fired → weaken all incoming synapses
        # dW[pre, post] -= A_minus * trace_pre[pre]
        ltd = (spikes.unsqueeze(0) *
               self.trace_pre.unsqueeze(1) *
               p.A_minus)

        dW = (ltp - ltd) * self.mask.float()

        self.W = (self.W + dW).clamp(p.w_min, p.w_max)
        self.W = self.W * self.mask.float()   # zero out non-synapses

        # Age existing synapses
        self.age = self.age + self.mask.float() * dt_ms

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
        self.g_e = torch.zeros(N, device=device)   # excitatory
        self.g_i = torch.zeros(N, device=device)   # inhibitory

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
        self.g_e = self.g_e * torch.exp(torch.tensor(-dt_ms / p.tau_syn_e, device=self.device))
        self.g_i = self.g_i * torch.exp(torch.tensor(-dt_ms / p.tau_syn_i, device=self.device))

        # ── Membrane integration ──
        # Modulate tau_m by serotonin (calm = slower integration)
        tau_m_eff = p.tau_m * mod.serotonin

        # Leak current
        I_leak = (p.V_rest - self.V) / tau_m_eff

        # Synaptic current (net = excitatory - inhibitory)
        I_syn = (self.g_e - self.g_i) / p.C_m

        # Thermal / background noise (norepinephrine scales arousal noise)
        I_noise = torch.randn(self.N, device=self.device) * p.noise_std * mod.norepinephrine

        # Total integration
        dV = (I_leak + I_syn + I_ext + I_noise) * dt_ms
        self.V = torch.where(in_ref, torch.tensor(p.V_reset, device=self.device), self.V + dV)

        # ── Spike detection ──
        # Threshold modulated by norepinephrine (arousal lowers threshold)
        V_thresh_eff = p.V_thresh - (mod.norepinephrine - 1.0) * 3.0
        spikes = (self.V >= V_thresh_eff) & (~in_ref)

        # ── Post-spike reset ──
        self.V        = torch.where(spikes, torch.tensor(p.V_reset, device=self.device), self.V)
        self.ref_count = torch.where(spikes,
                                     torch.tensor(p.t_ref, device=self.device),
                                     self.ref_count)

        # ── Propagate spikes via synapses ──
        # Each firing neuron injects current scaled by its sign (Dale's Law)
        # into all its postsynaptic targets
        spike_f = spikes.float() * self.sign          # [N], signed
        # I_post[j] = sum_i W[i,j] * spike_f[i]
        I_post  = torch.matmul(spike_f, synapses.W * synapses.mask.float())  # [N]

        # Separate into excitatory and inhibitory
        exc_in = I_post.clamp(min=0)
        inh_in = (-I_post).clamp(min=0)
        self.g_e = self.g_e + exc_in
        self.g_i = self.g_i + inh_in

        # ── Update spike history ──
        self.spike_history[:, self.history_ptr] = spikes
        self.history_ptr = (self.history_ptr + 1) % self.spike_history.size(1)

        # ── Update firing rate EMA ──
        self.firing_rate = 0.99 * self.firing_rate + 0.01 * spikes.float()

        return spikes


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
        if self.step_count % (p.prune_interval // p.seek_interval) == 0:
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
        G = (C > self.threshold).cpu().numpy()

        # Connected components via BFS
        visited    = [False] * self.N
        assemblies = []

        for start in range(self.N):
            if visited[start] or not G[start].any():
                continue
            component = []
            queue     = [start]
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                neighbors = np.where(G[node])[0]
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
                 seed: int = 42):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.N      = N
        self.params = params or NeuronParams()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t      = 0        # simulation time (ms)
        self.step_i = 0        # step counter

        print(f"[BrainNet] Initializing {N} neurons on {self.device}")

        # Neuromodulator state (can be set externally)
        self.modulator = NeuromodulatorState()

        # Core components
        self.population = NeuronPopulation(N, self.params, self.modulator, self.device)
        self.synapses   = SynapseMatrix(N, self.params, self.device)
        self.growth     = GrowthCone(self.params, self.modulator, self.device)
        self.thoughts   = ThoughtDetector(N, device=self.device)

        # ── Seed initial sparse random connectivity (like newborn brain) ──
        # ~5% connection probability, random weights
        init_prob = 0.05
        n_init    = int(N * N * init_prob)
        pre_idx   = torch.randint(0, N, (n_init,))
        post_idx  = torch.randint(0, N, (n_init,))
        for pre, post in zip(pre_idx.tolist(), post_idx.tolist()):
            if pre != post:
                self.synapses.add_synapse(pre, post)

        print(f"[BrainNet] Initial synapses: {self.synapses.mask.sum().item()}")
        print(f"[BrainNet] Inhibitory neurons: {self.population.is_inhibitory.sum().item()}")
        print("[BrainNet] Ready.\n")

        # Metrics log
        self._log: Dict[str, List] = {
            'time'         : [],
            'n_spikes'     : [],
            'n_synapses'   : [],
            'n_thoughts'   : [],
            'thought_size' : [],
            'stability'    : [],
            'firing_rate'  : [],
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

        # ── LIF integration + spike detection ──
        spikes = self.population.step(I_ext, self.synapses, dt)

        # ── STDP weight updates ──
        self.synapses.apply_stdp(spikes.float(), dt, self.modulator)

        # ── Growth cone sweep (periodic) ──
        new_synapses = 0
        if self.step_i % self.params.seek_interval == 0:
            new_synapses = self.growth.sweep(self.population, self.synapses)

        # ── Thought detection (periodic) ──
        assemblies = []
        if self.step_i % 25 == 0:
            assemblies = self.thoughts.detect(self.population.spike_history)

        # ── Log ──
        if self.step_i % 10 == 0:
            self._log['time'].append(self.t)
            self._log['n_spikes'].append(spikes.sum().item())
            self._log['n_synapses'].append(self.synapses.mask.sum().item())
            self._log['n_thoughts'].append(len(assemblies))
            self._log['thought_size'].append(
                np.mean([len(a) for a in assemblies]) if assemblies else 0
            )
            self._log['stability'].append(self.thoughts.assembly_stability())
            self._log['firing_rate'].append(self.population.firing_rate.mean().item())

        self.t      += dt * 1000.0   # advance time in ms
        self.step_i += 1

        state = {
            'time_ms'       : self.t,
            'n_spikes'      : int(spikes.sum().item()),
            'n_synapses'    : int(self.synapses.mask.sum().item()),
            'n_thoughts'    : len(assemblies),
            'assemblies'    : assemblies,
            'thought_stability': self.thoughts.assembly_stability(),
            'new_synapses'  : new_synapses,
            'mean_firing_hz': float(self.population.firing_rate.mean().item()) / dt,
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
        Release dopamine (reward signal).
        Scales LTP — neurons that recently fired get their connections strengthened.
        value  : dopamine level (1.0 = baseline, >1 = reward, <1 = punishment)
        decay  : how fast it returns to baseline per step
        """
        self.modulator.dopamine = value
        # Schedule decay (handled externally in step loop if desired)

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

    def connectivity_stats(self) -> dict:
        n_syn   = int(self.synapses.mask.sum().item())
        w_mean  = float(self.synapses.W[self.synapses.mask].mean().item()) if n_syn > 0 else 0
        w_std   = float(self.synapses.W[self.synapses.mask].std().item())  if n_syn > 0 else 0
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
        """Save full network state."""
        torch.save({
            'V'             : self.population.V,
            'W'             : self.synapses.W,
            'mask'          : self.synapses.mask,
            'memory1'       : self.population.firing_rate,
            'modulator'     : self.modulator.__dict__,
            't'             : self.t,
            'step_i'        : self.step_i,
            'params'        : self.params.__dict__,
        }, path)
        print(f"[BrainNet] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.population.V         = ckpt['V']
        self.synapses.W           = ckpt['W']
        self.synapses.mask        = ckpt['mask']
        self.t                    = ckpt['t']
        self.step_i               = ckpt['step_i']
        print(f"[BrainNet] Loaded from {path}")
