# BioNeuron — Test Results

Every mechanism in this library was verified against a known-correct expectation
before being committed. This document records those tests and their results.
The fast checks are reproducible via `python tests/test_bioneuron.py` (5/5
passing); the longer learning runs are reproduced by the `demo_*` functions in
`closed_loop.py`. All numbers below are from CPU runs unless noted.

> **Architectural note.** `BrainNet` exposes two interchangeable backends behind a
> uniform API (see *System Architecture* in the [README](../README.md)): the
> connectivity backend (dense `SynapseMatrix` ↔ edge-list `SparseSynapses`) and
> the neuromodulator (`NeuromodulatorState` ↔ full `Neurochemistry`). §3 and §5
> below are precisely the regression tests that pin those two swaps as
> behaviourally equivalent — they are what makes the pluggability safe.

---

## 1. Core correctness

| Test | Method | Result |
|---|---|---|
| **STDP causality** (Bi & Poo 1998) | Drive a 2-neuron synapse pre→post then post→pre | pre→post **+0.0095 (LTP)**, post→pre **−0.0114 (LTD)** ✓ — fixed a pre-existing sign inversion (v1.1) |
| STDP trace ordering | Traces registered after the weight update | correct nearest-/all-to-all pair rule ✓ |
| Prune cadence guard | `prune_interval < seek_interval` | no longer divides by zero ✓ |

## 2. Biophysics (v1.1–v1.2)

| Test | Expectation | Result |
|---|---|---|
| Spike-frequency adaptation | Lowers steady firing under constant drive | 1318 → **1189** spikes (on < off) ✓ |
| √dt noise scaling | Modest noise → spontaneous activity | silent at `noise_std`=1.2 with `dt`-scaling; **fires** with √dt scaling ✓ |
| Oscillation analyzer | Band powers sum to ~1 in-range; identify dominant band | sum ≈ 1.00, dominant band returned ✓ |
| Short-term plasticity (Tsodyks–Markram) | Depression falls, facilitation `u` rises across a train | depression **0.75 → 0.013**; facilitation `u` **0.28 → 0.67** ✓ |
| NMDA voltage gate (Jahr & Stevens) | Blocked at rest, open when depolarised | B(−70 mV) = **0.044**, B(0 mV) = **0.781** ✓ |
| Criticality branching ratio σ | σ=1 for constant activity, 0.8 for 0.8-decay | **1.000** and **0.798** ✓ |
| Spike statistics (CV of ISI) | ~0 clockwork, ~1 Poisson | **0.00** and **0.88** ✓ |
| Opt-in features run | conductance synapses, delays, homeostasis | all run without error ✓ |

## 3. Sparse backend equivalence & scaling (v1.3)

| Test | Result |
|---|---|
| `propagate()` dense vs sparse | match to **5e-7** (float reduction order) ✓ |
| `apply_stdp()` per-edge weights | **exact, 0.0 difference** ✓ |
| Full spike trains, matched connectivity | **100 % identical** over 150 steps ✓ |
| Scale | N=3000 / 439 k edges in **7 MB** vs **72 MB** dense; dense `[N,N]` would be ~512 MB at N=8000 ✓ |
| Side benefit | vectorized bulk init replaced the per-synapse Python loop (faster dense init too) |

## 4. Reward / stress learning (v1.4)

Three-factor plasticity: eligibility traces gated by dopamine (reward) / cortisol (stress). Verified identically on **both** backends.

| Condition | A→A pathway weight |
|---|---|
| No teaching signal | 0.049 → 0.049 (no drift — learning needs a signal) ✓ |
| **Reward** the active pathway | 0.049 → **0.295** (potentiated) ✓ |
| **Cortisol** the active pathway | 0.049 → **0.000** (depressed) ✓ |
| Idle pathway during reward | 0.048 → 0.048 (untouched — clean credit assignment) ✓ |

*Finding:* the eligibility tag must be the **causal LTP component**, not the net `(LTP−LTD)` — with `A_minus > A_plus`, the net tag is negative under synchronous co-activation, which sign-flips reward/punishment. Caught and fixed in testing.

## 5. Neurochemistry system (v1.5)

| Test | Result |
|---|---|
| Drop-in for `NeuromodulatorState` | axis read/write compatible ✓ |
| Histamine → arousal axis | 1.00 → **1.30** ✓ |
| Adenosine (sleep pressure) → arousal/attention | both drop below 0.95 ✓ |
| Oxytocin buffers cortisol | cortisol axis reduced ✓ |
| Kinetics | dopamine clears fast (2.0→1.08 in 500 ms), cortisol slow (2.0→1.87 in 200 ms) ✓ |
| Reward learning through the chemistry | 0.049 → **0.295** ✓ |
| Custom chemical (caffeine) | registers and pushes axes ✓ |
| Default path | unchanged ✓ |

## 6. Closed-loop learning from self-generated feedback (v1.5)

The network's *own* decoded output decides correct/wrong; `feedback()` is applied automatically.

| Demo | Result |
|---|---|
| **Operant conditioning** (`demo_conditioning`, Fetz 1969) | response rate **0.27 → 1.00** — learns to produce the target output on cue (reward-only positive reinforcement) ✓ |
| **2-AFC discrimination** (`demo_discrimination`) | **learns to ~0.85** mid-run with winner-take-all + reward-prediction-error + an explicit plastic readout; chance (~0.50) without any one of them |

**Two findings, both biologically meaningful:**
1. **Chronic cortisol cancels reward.** Since the learning signal is `M = (dopamine−1) − cortisol` and cortisol clears slowly, frequent errors keep stress elevated and null the next reward — a realistic "chronic stress impairs reward learning" effect. Conditioning therefore uses positive reinforcement only.
2. **Discrimination needs three ingredients together:** (a) winner-take-all lateral inhibition so only the winner is eligible, (b) a dopamine reward-*prediction-error* baseline (not raw reward) so correct/wrong don't cancel, and (c) a dedicated plastic input→output projection so the credit isn't buried in recurrent noise. With all three it climbs to ~0.85, then shows a **late instability** typical of unregulated R-STDP (positive feedback + weight saturation). *Next step:* add homeostatic regulation (`enable_homeostasis=True`) to clamp the runaway.

Representative discrimination curve (per-batch accuracy):
`0.53 → 0.63 → 0.70 → 0.63 → 0.80 → 0.87 → 0.87 → 0.83 → 0.40(collapse)`

## 7. Packaging (v1.5)

| Test | Result |
|---|---|
| Flat imports (notebook style) | still work ✓ |
| Editable install `pip install -e ohToBeAlive` | succeeds ✓ |
| `import bioneuron` + full public API | v1.5.0, all names import ✓ |
| Console entry point `bioneuron version` | works ✓ |
| `python -m build` | builds **`bioneuron-1.5.0-py3-none-any.whl`** + sdist ✓ (PyPI-ready) |

## 8. Performance benchmark (sparse backend, dt = 0.1 ms)

Real-time = 10,000 steps/s. Hardware: this CPU and an **RTX 3070 Laptop GPU (8.6 GB)**.

| Device | N | E | steps/s | % real-time |
|---|---|---|---|---|
| CPU | 1,000 | 49 k | 261 | 2.6 % |
| CPU | 3,000 | 439 k | 29 | 0.29 % |
| RTX 3070 | 3,000 | 439 k | 73 | 0.73 % |
| RTX 3070 | 10,000 | 4.9 M | 7.5 | 0.08 % |
| RTX 3070 | 30,000 | 43.9 M | 0.7 | 0.01 % |

Full hardware / scale analysis (including the "bare consciousness" question) is in [`HARDWARE.md`](HARDWARE.md). Headline: throughput is bottlenecked by per-step Python overhead, not the GPU — the compute core needs CUDA-kernel fusion before hardware becomes the limit.

## 9. Structured cortex & gamma (v1.6)

`cortex.py` adds cell-type heterogeneity (per-neuron `tau_m`, `t_ref`, `b_adapt`, `V_thresh`) and the canonical E–I microcircuit. The biological payoff is **gamma via PING** (pyramidal–interneuron network gamma; Cardin et al. 2009).

| Test | Expectation | Result (N=800, driven) |
|---|---|---|
| Cell-type dissociation | FS interneurons out-fire RS pyramidal cells | **FS 160 Hz vs RS 54 Hz** ✓ |
| PING rhythm | A gamma-band population oscillation from the E–FS loop | **dominant band = gamma** (power 0.53), peak ≈ 24–30 Hz (high-beta/low-gamma) ✓ |
| Default path | Homogeneous network unaffected by per-neuron override machinery | 6/6 unit tests pass ✓ |

The rhythm sits at the beta/gamma border (~24–30 Hz) rather than mid-gamma; frequency rises with faster inhibition (`tau_syn_i`) and conduction delay. FS interneurons reaching ~160 Hz is the characteristic PV⁺ fast-spiking signature. This is the first emergent dynamic that a random homogeneous network cannot produce — it requires the distinct fast, non-adapting interneuron class wired into feedback inhibition.

## 10. Neural-fidelity upgrades (v1.7)

Three canonical mechanisms, all opt-in, all verified (unit tests added; 8/8 pass).

| Mechanism | Reference | Test | Result |
|---|---|---|---|
| **AdEx** exponential spike initiation | Brette & Gerstner 2005 | Fires, stays numerically bounded, adaptation modulates rate | fires & stable (no NaN/Inf); b=0 → 902 vs b=0.15 → 774 spikes ✓ |
| **Inhibitory STDP** | Vogels et al. 2011 | Suppresses runaway excitation toward `rho_target` | ON: early 805 → late **519**; OFF: 859 → 778 ✓ |
| **Gap junctions** (electrical synapses) | Galarreta & Hestrin 1999 | Increase FS-interneuron synchrony | FS population-rate CV 2.57 → **2.70** (gap on) ✓ |

iSTDP is the recommended stabilizer for the closed-loop discrimination instability (§6): inhibition learns to clamp the excitatory pathways the reward signal over-potentiates. AdEx + adaptation parameters span the cortical firing-pattern repertoire (Naud et al. 2008). Gap junctions among FS cells tighten the interneuron network that drives PING gamma (§9). The default (homogeneous, AdEx/iSTDP/gap off) path is unchanged.

## 11. Dendritic computation (v1.8)

A two-compartment neuron (`enable_dendrite`): excitation (and NMDA) charges a separate **active dendrite** with regenerative plateau/dendritic spikes, coupled to the soma; inhibition stays perisomatic. A single neuron becomes a two-layer computation (Poirazi & Mel 2003; Larkum 2013).

| Test | Expectation | Result |
|---|---|---|
| Supralinear summation | Combined input evokes *more* than the sum of its parts (dendritic-spike signature) | group A → 4 spikes, group B → 4, A+B → **10 (not 8)**, ratio **1.25** ✓ |
| Point-neuron contrast | A point neuron cannot amplify the same distributed sub-threshold input | point neuron fires **0** to input the dendrite turns into 10 ✓ |
| Default path | Single-compartment dynamics unchanged when off | 9/9 unit tests pass ✓ |

This is the biggest single-neuron realism upgrade: dendrites let a neuron detect *clustered* synaptic input that a point neuron ignores entirely, and sum it supralinearly — the substrate for the "neuron as a two-layer network" view of cortical computation.
