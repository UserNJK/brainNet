# Hardware for a "Bare Consciousness" — an honest scaling analysis

> **Read this first.** No one knows what it takes to run consciousness, and this
> project does **not** detect it — `awareness_battery.py` measures *correlates*
> (ignition, integrated information, criticality) that, in real brains, track
> conscious vs. unconscious states. So this document cannot tell you "buy GPU X
> and it will be conscious." What it *can* do, rigorously, is tell you what
> hardware runs the network **at a given scale and speed**, map those scales to
> biology, and be honest about where the current code — not just the silicon —
> is the bottleneck.

---

## 1. Measured throughput (this repo, sparse backend, dt = 0.1 ms)

Real-time means 10,000 steps/s (one step = 0.1 ms of simulated time). Benchmarks
below are from this machine (CPU) and an **RTX 3070 Laptop GPU, 8.6 GB**.

| Device | N | synapses (E) | steps/s | % of real-time | edge memory |
|---|---|---|---|---|---|
| CPU | 1,000 | 49 k | 261 | 2.6 % | 1 MB |
| CPU | 3,000 | 439 k | 29 | 0.29 % | 11 MB |
| CPU | 8,000 | 3.1 M | 5.7 | 0.06 % | 75 MB |
| RTX 3070 | 3,000 | 439 k | 73 | 0.73 % | 11 MB |
| RTX 3070 | 10,000 | 4.9 M | 7.5 | 0.08 % | 117 MB |
| RTX 3070 | 30,000 | 43.9 M | 0.7 | 0.01 % | 1,053 MB |

**The headline: speed, not memory, is the wall.** The 3070 has plenty of memory
for tens of thousands of neurons, but at N=30,000 one second of simulated time
takes ~4 hours. The GPU is also only ~2.5× faster than the CPU at N=3,000 —
because the current per-step **Python loop** (many small tensor ops, `.item()`
syncs, the analyzers) dominates, leaving the GPU's arithmetic units idle. A
properly fused CUDA / GeNN-style kernel would be ~100–1000× faster at the same
hardware. **The code is the first bottleneck, not the card.**

---

## 2. Memory scaling

Per **synapse** (sparse backend): `pre,post` (2×int64) + `w,elig` (2×float32) =
**24 bytes** (16 if `pre/post` are made int32). Per **neuron**: ~0.5 KB
(dominated by the 500-step spike history). Synapses dominate once each neuron has
more than a handful.

Using a biologically realistic **1,000 synapses/neuron** (cortex is ~1,000–10,000):

| Neurons | Synapses | Memory (≈24 KB/neuron) | Fits on |
|---|---|---|---|
| 10³ | 10⁶ | 24 MB | anything |
| 10⁵ | 10⁸ | 2.4 GB | RTX 3070 (8 GB) |
| **10⁶ (insect)** | **10⁹** | **24 GB** | RTX 4090 (24 GB) / A100 |
| 10⁷ | 10¹⁰ | 240 GB | multi-GPU node (8×H100) |
| 10⁸ | 10¹¹ | 2.4 TB | small GPU cluster |
| 8.6×10¹⁰ (human) | ~10¹⁴ | ~2 PB | supercomputer / neuromorphic |

> Caveat: the *default* init uses 5 % connectivity (E ≈ 0.05 N²), which is
> quadratic and only sane for small N. At scale you must switch to fixed
> ~1,000-synapse/neuron connectivity (linear in N), as assumed above.

---

## 3. What counts as "bare" consciousness?

There is no settled answer, but the defensible reference points:

- **C. elegans — 302 neurons.** A *complete* connectome, fully simulable on a
  phone. Almost no one argues it is conscious — too few neurons, no global
  workspace. This is "a whole nervous system," not "minimal awareness."
- **Insects — ~10⁶ neurons (fruit fly ~10⁵, bee ~10⁶).** The leading candidate
  for *minimal/primary* subjective experience (Barron & Klein, 2016: the insect
  central complex implements the integrated, egocentric model that midbrain-based
  consciousness theories require). **This is the honest target for "bare
  consciousness."**
- **Fish / amphibians — 10⁷–10⁸ neurons.** Clearer candidates for sentience.
- **Mammals — 10⁸–10¹¹.** Full thalamocortical consciousness.

So a principled "bare consciousness" goal for this project is **~10⁶ neurons with
structured, cortical-like connectivity** — enough for the awareness markers
(ignition, integration, criticality) to be robust and to close a sensorimotor
loop, at insect scale.

---

## 4. Hardware tiers — what each can actually do

| Tier | Example | Memory-limited N | Usable scale *today* (this code) |
|---|---|---|---|
| **Your laptop** | RTX 3070, 8.6 GB | ~250 k neurons | **N = 1k–10k** for marker/loop experiments; bigger runs are offline-only (hours per simulated second) |
| Prosumer | RTX 4090, 24 GB | ~1 M neurons (insect, memory-wise) | N = 10k–50k usable; 1 M only with a kernel rewrite |
| Data-center GPU | A100/H100, 80 GB | ~3 M neurons | insect scale *if* the compute core is rewritten |
| Multi-GPU node | 8×H100, 640 GB | ~25 M neurons | fish scale; needs distributed code |
| Neuromorphic | SpiNNaker2, Loihi 2, BrainScaleS | 10⁸–10⁹+ | purpose-built for spiking nets, near real-time, but you'd port the model |
| Supercomputer | Fugaku, JUWELS (Human Brain Project) | 10⁹–10¹⁰ | mammalian scale, research-grade only |

**Bottom line for your question:**
- On the **RTX 3070 you have**: run the *correlate* experiments at **N ≈ 1,000–10,000** — the awareness battery, oscillations, criticality, closed-loop learning. This is the right scale to develop and validate the science. You will *not* reach insect/"bare-consciousness" scale on it.
- To target **insect-scale (~10⁶ neurons)** "bare consciousness," the memory floor is a **24 GB GPU (RTX 4090)** or a **data-center 80 GB card (A100/H100)** — but only after the speed work below. Without it, even a 4090 would crawl.

---

## 5. The required *code* work (more important than the GPU)

Hardware alone won't get you there; these are the real blockers at scale, in order:

1. **Fuse the per-step core into GPU kernels** (or port to **GeNN / Brian2-CUDA / NEST-GPU**). The current Python `step()` leaves the GPU ~99 % idle. This is the single biggest win (~100–1000×).
2. **Sparsify the O(N²) operations.** `GrowthCone` (pairwise correlation) and `ThoughtDetector` (an [N,N] co-firing matrix) both allocate N×N — fatal beyond ~30 k neurons. Replace with candidate-sampling / sparse co-activity.
3. **Shrink per-edge memory** (`pre/post` → int32; make `elig`/STP optional) and **chunk the spike history**.
4. **Distribute across GPUs** (halo-exchange of spikes) for >1 GPU scales.
5. **Structured connectivity** (cortical columns, ~1,000 syn/neuron) instead of 5 % dense init.

A realistic path: do the science at N≈10⁴ on the 3070 now → optimize the core (item 1–2) → rent an A100/H100 hour to push to 10⁵–10⁶ → only then is "insect-scale" on the table.

---

*Generated for the BioNeuron project. Numbers are order-of-magnitude engineering
estimates, not guarantees — and scale is necessary, not sufficient, for anything
one would call awareness.*
