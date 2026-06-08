"""
closed_loop.py — Natural feedback for BrainNet
==============================================
Turns the hand-labelled `brain.feedback(correct)` call into a *closed loop*: the
network's OWN output decides whether it was right, and reward (dopamine) or
stress (cortisol) is applied automatically. Nothing external tells the network
the answer step-by-step — it acts, its output is read out, the readout is scored
against the task, and the resulting reward/punishment shapes the very synapses
that produced the action (reward-modulated STDP / R-STDP; Izhikevich 2007,
Legenstein et al. 2008).

Anatomy of a trial:
    1. ENCODE   — a stimulus drives a group of "sensory" input neurons
    2. ACT      — the network runs; "motor" output groups accumulate spikes
    3. DECODE   — the most active output group is the network's choice
    4. SCORE    — choice vs. the task's correct answer → correct / wrong
    5. FEEDBACK — brain.feedback(correct): dopamine if right, cortisol if wrong
    6. CONSOLIDATE — a short settle window lets the modulator gate the eligible
                     synapses (the pathway that just fired)

Requires a BrainNet built with `enable_reward_learning=True` for the feedback to
actually change weights.

    from core import BrainNet, NeuronParams
    from closed_loop import ClosedLoop

    p = NeuronParams(enable_reward_learning=True, noise_std=2.0, noise_sqrt_dt=True)
    brain = BrainNet(N=200, params=p)
    loop  = ClosedLoop(brain,
                       input_groups={0: range(0,20),  1: range(20,40)},
                       output_groups=[range(40,60), range(60,80)])
    log = loop.run(n_trials=300)        # learns to map input c -> output c
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Optional, Callable
import numpy as np
import torch


class ClosedLoop:
    """
    Wraps a BrainNet with sensory inputs and motor outputs and runs scored
    trials, applying brain.feedback() from the network's own decoded choice.
    """

    def __init__(self, brain,
                 input_groups: Dict[int, Sequence[int]],
                 output_groups: List[Sequence[int]],
                 dt: float = 1e-4,
                 stim_amp: float = 5.0,
                 stim_steps: int = 60,
                 response_steps: int = 40,
                 settle_steps: int = 100,
                 reward_mag: float = 1.0,
                 warmup_steps: int = 200,
                 wta_strength: float = 0.0,
                 use_cortisol: bool = False,
                 reward_baseline: bool = False):
        self.brain   = brain
        self.dt      = dt
        self.inputs  = {int(k): list(v) for k, v in input_groups.items()}
        self.outputs = [list(g) for g in output_groups]
        self.stim_amp       = stim_amp
        self.stim_steps     = stim_steps
        self.response_steps = response_steps
        self.settle_steps   = settle_steps
        self.reward_mag     = reward_mag
        self.warmup_steps   = warmup_steps
        # Winner-take-all: each output group is inhibited in proportion to the
        # activity of its competitors (lateral/feedback inhibition), so only the
        # winning group stays active — and thus only its pathway is tagged
        # eligible, letting reward/punishment differentiate the choices.
        self.wta_strength = wta_strength
        self.use_cortisol = use_cortisol
        # Reward baseline: dopamine should encode a reward *prediction error*
        # (outcome − expected), not raw outcome. This concentrates learning on
        # surprising trials and removes the symmetric reward/punishment that
        # otherwise cancels out — the key to discrimination learning.
        self.reward_baseline = reward_baseline
        self._expected = 0.5          # running estimate of P(correct)
        self._dev    = brain.device
        self._warmed = False

    def warmup(self):
        """Run the network to its baseline regime before scored trials begin."""
        for _ in range(self.warmup_steps):
            self.brain.step(dt=self.dt)
        self._warmed = True

    # ── one trial ─────────────────────────────────────────────────
    def present(self, stim_class: int):
        """Drive the stimulus, accumulate output-group spikes, decode the choice.
        With wta_strength>0, competing output groups inhibit each other so the
        leading group wins (and is the only one left eligible for credit)."""
        N = self.brain.N
        counts = np.zeros(len(self.outputs))
        act = np.zeros(len(self.outputs))               # fast per-group activity trace
        stim = torch.zeros(N, device=self._dev)
        stim[self.inputs[stim_class]] = self.stim_amp

        def run(use_stim: bool):
            I = stim.clone() if use_stim else torch.zeros(N, device=self._dev)
            if self.wta_strength > 0 and len(self.outputs) > 1:
                total = act.sum()
                for c, g in enumerate(self.outputs):
                    I[g] = I[g] - self.wta_strength * (total - act[c])   # inhibited by rivals
            sp = self.brain.step(I, dt=self.dt)[0].detach().to('cpu').numpy()
            for c, g in enumerate(self.outputs):
                ga = sp[g].sum()
                counts[c] += ga
                act[c] = 0.8 * act[c] + 0.2 * ga

        for _ in range(self.stim_steps):                 # ENCODE + ACT
            run(True)
        for _ in range(self.response_steps):             # ACT (stimulus off)
            run(False)

        choice = int(np.argmax(counts)) if counts.sum() > 0 else -1
        return choice, counts

    def trial(self, stim_class: int, target_class: Optional[int] = None) -> dict:
        """Present a stimulus, decode, score, and apply self-generated feedback."""
        if target_class is None:
            target_class = stim_class
        choice, counts = self.present(stim_class)
        correct = (choice == target_class)
        if self.reward_baseline:
            # Dopamine = baseline + reward-prediction-error (Schultz 1997).
            outcome = 1.0 if correct else 0.0
            rpe = outcome - self._expected
            self._expected += 0.05 * rpe                 # slow value estimate
            self.brain.modulator.dopamine = max(0.0, 1.0 + 2.0 * self.reward_mag * rpe)
        else:
            self.brain.feedback(correct, magnitude=self.reward_mag,
                                use_cortisol=self.use_cortisol)    # ← the natural feedback
        for _ in range(self.settle_steps):               # CONSOLIDATE (modulator gates eligibility)
            self.brain.step(dt=self.dt)
        return dict(stim=stim_class, target=target_class,
                    choice=choice, correct=bool(correct), counts=counts.tolist())

    # ── a session ─────────────────────────────────────────────────
    def run(self, n_trials: int,
            classes: Optional[Sequence[int]] = None,
            target_fn: Optional[Callable[[int], int]] = None,
            rng: Optional[np.random.Generator] = None,
            verbose: bool = True) -> List[dict]:
        """
        Run `n_trials` of randomly-drawn stimuli. `target_fn(stim)` gives the
        correct class (defaults to identity: learn to echo the stimulus).
        """
        rng = rng or np.random.default_rng(0)
        classes = list(classes) if classes is not None else list(self.inputs)
        if not self._warmed:
            self.warmup()
        log = []
        for i in range(n_trials):
            c = int(rng.choice(classes))
            tgt = target_fn(c) if target_fn else c
            r = self.trial(c, tgt)
            log.append(r)
            if verbose and (i + 1) % max(1, n_trials // 10) == 0:
                w = log[-max(1, n_trials // 10):]
                acc = np.mean([x['correct'] for x in w])
                print(f"  trial {i+1:4d}/{n_trials} | recent accuracy {acc:.2f}")
        return log


def accuracy_curve(log: List[dict], window: int = 20) -> np.ndarray:
    """Sliding-window accuracy over a trial log."""
    c = np.array([x['correct'] for x in log], dtype=float)
    if len(c) < window:
        return c.cumsum() / (np.arange(len(c)) + 1)
    return np.convolve(c, np.ones(window) / window, mode='valid')


# ═════════════════════════════════════════════════════════════════
#  Demo task: two-way stimulus→response discrimination, learned purely
#  from the network's own feedback.
# ═════════════════════════════════════════════════════════════════

def demo_discrimination(n_trials: int = 300, N: int = 160, seed: int = 0,
                        device=None, wta_strength: float = 0.35, verbose: bool = True):
    """
    Two stimuli, two response groups. The network must route stimulus 0 → output
    0 and stimulus 1 → output 1. It is never told the mapping — it only ever
    receives a reward-prediction-error (dopamine burst when its own chosen output
    matches, dip when it does not). Winner-take-all lateral inhibition between the
    output groups makes the choice (and the eligible pathway) unambiguous, which
    is what lets the credit signal separate the two mappings. Returns
    (early_acc, late_acc, log).
    """
    try:
        from .core import BrainNet, NeuronParams
    except ImportError:
        from core import BrainNet, NeuronParams
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = NeuronParams(enable_reward_learning=True, noise_std=2.0, noise_sqrt_dt=True,
                     seek_interval=10**9, prune_interval=10**9,
                     neuromod_lr=0.08, tau_elig=300.0, w_max=2.0)
    brain = BrainNet(N=N, params=p, device=device, seed=seed)
    brain.synapses.W *= 2.2     # responsive regime so inputs drive the outputs
    loop = ClosedLoop(
        brain,
        input_groups={0: range(0, 20), 1: range(20, 40)},
        output_groups=[range(40, 70), range(70, 100)],
        stim_amp=6.0, stim_steps=60, response_steps=60, settle_steps=120,
        reward_mag=1.0, warmup_steps=300, wta_strength=wta_strength,
        reward_baseline=True,      # dopamine = reward-prediction-error (see ClosedLoop)
    )
    log = loop.run(n_trials, verbose=verbose)
    acc = np.array([x['correct'] for x in log], dtype=float)
    third = max(1, n_trials // 3)
    early, late = float(acc[:third].mean()), float(acc[-third:].mean())
    return early, late, log


def demo_conditioning(n_trials: int = 150, N: int = 120, seed: int = 0,
                      device=None, verbose: bool = True):
    """
    Operant conditioning of neural activity (Fetz 1969). A stimulus is presented
    and the network is rewarded whenever its OWN target output group crosses an
    activity threshold — otherwise it gets cortisol. The feedback is entirely
    self-generated (it depends only on what the network did), and reward-gated
    plasticity ratchets up the rewarded pathway, so the network learns to
    produce the target response on cue.

    Returns (early_rate, late_rate, log) where rate = fraction of trials the
    network produced the rewarded response.
    """
    try:
        from .core import BrainNet, NeuronParams
    except ImportError:
        from core import BrainNet, NeuronParams
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = NeuronParams(enable_reward_learning=True, noise_std=2.0, noise_sqrt_dt=True,
                     seek_interval=10**9, prune_interval=10**9,
                     neuromod_lr=0.05, tau_elig=400.0, w_max=2.0)
    brain = BrainNet(N=N, params=p, device=device, seed=seed)
    brain.synapses.W *= 2.2
    S = list(range(0, 20)); O = list(range(N - 30, N))     # stimulus, target output
    dt = 1e-4

    def present():
        cnt = 0.0
        stim = torch.zeros(N, device=device); stim[S] = 6.0
        for _ in range(60):
            cnt += brain.step(stim, dt=dt)[0].detach().to('cpu').numpy()[O].sum()
        for _ in range(40):
            cnt += brain.step(dt=dt)[0].detach().to('cpu').numpy()[O].sum()
        return cnt

    for _ in range(300):                                   # warmup
        brain.step(dt=dt)
    probes = [present() for _ in range(12)]                # calibrate threshold
    threshold = float(np.median(probes))

    log = []
    for i in range(n_trials):
        responded = present() >= threshold
        # Reward-only positive reinforcement: reward when the network produced the
        # target response, do nothing otherwise. (Punishing misses would pin the
        # slow cortisol level high, and since the learning signal is
        # M = (dopamine−1) − cortisol, lingering stress cancels the next reward —
        # a realistic "chronic stress blocks reward learning" effect, but it
        # stalls simple conditioning. See demo_discrimination's note.)
        if responded:
            brain.reward(2.0)                              # self-generated reward
        for _ in range(120):
            brain.step(dt=dt)
        log.append(bool(responded))
        if verbose and (i + 1) % max(1, n_trials // 10) == 0:
            w = log[-max(1, n_trials // 10):]
            print(f"  trial {i+1:4d}/{n_trials} | recent response rate {np.mean(w):.2f}")
    acc = np.array(log, dtype=float)
    third = max(1, n_trials // 3)
    return float(acc[:third].mean()), float(acc[-third:].mean()), log


if __name__ == '__main__':
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else 'conditioning'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    if task == 'discrimination':
        early, late, _ = demo_discrimination(n_trials=n)
        print(f"\nSelf-taught 2-way discrimination: {early:.2f} -> {late:.2f} (chance 0.50)")
    else:
        early, late, _ = demo_conditioning(n_trials=n)
        print(f"\nOperant conditioning (self-rewarded): response rate {early:.2f} -> {late:.2f}")
