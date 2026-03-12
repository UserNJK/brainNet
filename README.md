# BioNeuron — Biologically Accurate Spiking Neural Network

A from-scratch implementation of a cortical spiking neural network where each neuron operates on real biophysical principles — no backpropagation, no gradient descent. Learning is driven entirely by **Spike-Timing Dependent Plasticity (STDP)**, and connectivity is self-organized through **activity-dependent axon growth**.

---

## Core Architecture

**Leaky Integrate-and-Fire (LIF)**
Membrane dynamics with absolute refractory periods, excitatory/inhibitory synaptic conductances, and thermal noise injection. Dale's Law is enforced — 20% of neurons are GABAergic inhibitory, matching the actual cortical ratio.

**STDP Learning**
Follows the Bi & Poo (1998) rule: pre-before-post potentiates (LTP), post-before-pre depresses (LTD). No global error signal. No optimizer. Weights update purely from local spike timing.

**Growth Cone**
Each neuron periodically computes pairwise spike-train cross-correlations across the population and autonomously forms synapses toward neurons with correlated activity, modeling activity-dependent axon guidance. Synaptic pruning removes weights that decay below threshold, mirroring adolescent cortical pruning.

**Cell Assembly Detection**
Identifies emergent "thoughts" as synchronously firing neuron groups (Hebb, 1949) using co-firing correlation graphs and connected component analysis.

**Neuromodulator System**
Four globally-broadcast chemical signals:

| Modulator | Effect |
|---|---|
| `dopamine` | Scales LTP amplitude (reward-modulated STDP) |
| `acetylcholine` | Lowers synapse-formation threshold (attention / increased plasticity) |
| `serotonin` | Modulates membrane time constant (stability) |
| `norepinephrine` | Shifts firing threshold and noise amplitude (arousal) |

---

## What Makes This Different

Most "biologically inspired" networks are just MLPs with a renamed activation function. This network has no fixed topology — synapses are born, strengthened, and pruned entirely through dynamics. Thoughts are not labels or output logits; they are detected as **emergent synchrony patterns** within the network itself.

---

## Requirements

```bash
pip install torch numpy matplotlib
```

---

## Quickstart

```python
from core import BrainNet

brain = BrainNet(N=300)

for t in range(5000):
    spikes, state = brain.step(dt=1e-4)

brain.reward(2.0)            # dopamine burst
brain.attend(1.8)            # acetylcholine → aggressive synapse seeking
print(brain.get_thoughts())  # current cell assemblies
```

---

## References

- Hodgkin & Huxley (1952) — Action potential biophysics
- Bi & Poo (1998) — Synaptic modification by correlated activity (STDP)
- Hebb (1949) — Organization of Behavior (cell assemblies)
- Schultz et al. (1997) — Dopamine reward prediction error
- Dale (1935) — Pharmacology and nerve endings (Dale's Law)
