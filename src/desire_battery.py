"""
desire_battery.py — Operational Markers of DESIRE (incentive motivation)
=======================================================================

READ THIS FIRST — what this is and is NOT
-----------------------------------------
This battery does NOT claim to detect felt longing or subjective craving. As with
`awareness_battery.py`, no external measurement can confirm an inner state. What
it *does* do is measure the **operational markers of desire** that motivation
neuroscience uses to tell a system that *wants* an outcome from one that merely
reacts — and it validates them the only honest way: by checking that an INTACT,
dopamine-competent, contingently-trained network **dissociates** from control
conditions in which the machinery of wanting is disabled.

The governing idea is Berridge & Robinson's (1998) separation of "wanting"
(incentive salience — the dopamine-dependent motivational pull toward a
reward-predicting cue) from "liking" (hedonic impact, opioid-mediated). Desire is
*wanting*: it is learned, it is cue-triggered, it is dopamine-gated, and it is
directed at the specific object that caused reward. Those four properties give
four measurable markers, each grounded in a result the model already implements:

  1. Incentive-salience acquisition .. cue-evoked pursuit rises with rewarded
                                       training              (Berridge 1998)
  2. Pursuit asymptote ............... the trained level of cue-evoked pursuit
                                       ("wanting" magnitude)
  3. Reinstatement savings ........... after extinction, a single reward revives
                                       pursuit far faster than naive learning —
                                       a latent incentive memory (Bouton 2004)
  4. Directed credit ................. reward strengthens ONLY the pathway that
                                       caused it, not an unpaired one — desire is
                                       object-directed (Schultz 1997; Izhikevich 2007)

Substrate in BioNeuron. Desire here is the dopamine-gated three-factor pathway
(`enable_reward_learning`): recent co-active synapses are tagged eligible, and a
phasic dopamine signal (`brain.reward()`) converts the tag into potentiation
(M = da_gain·(dopamine−1) − cort_gain·cortisol). A cue→action projection that is
repeatedly active-then-rewarded is potentiated, so the cue acquires the power to
drive the action: incentive salience, mechanised.

Conditions (INTACT vs the two controls that kill wanting but not spiking):
  INTACT     — dopamine-competent, reward is contingent on the cue→action pathway.
  DA-LESION  — da_gain = 0: dopamine can no longer gate plasticity. Models
               dopamine depletion, which abolishes wanting while leaving
               consummatory "liking" and normal movement intact (Berridge 2007).
  DECOUPLED  — same amount of dopamine, delivered NON-contingently (during rest,
               unpaired from the cue→action pathway). Tests that it is the
               *contingency*, not the dopamine per se, that builds desire.

A clean result = INTACT tops every marker and the aggregate desire index, and both
controls collapse toward zero. Read it as: "this network shows the dynamical
signatures that, in animals, index incentive motivation" — NOT "it wants."

Hardware: CPU-friendly at N≈160 (the __main__ demo). device='cuda' for larger N.
"""

from __future__ import annotations
import numpy as np
import torch

try:
    from core import BrainNet, NeuronParams
except ImportError:                                   # installed-package layout
    from .core import BrainNet, NeuronParams


# ═════════════════════════════════════════════════════════════════
#  Network construction — a cue → action incentive pathway
# ═════════════════════════════════════════════════════════════════

def _params(da_gain: float = 1.0) -> NeuronParams:
    # Three-factor reward learning ON; structural growth/pruning OFF (we probe a
    # fixed architecture whose *weights* learn). Tuned to a lively, spontaneously
    # active regime so the cue can recruit the action group.
    return NeuronParams(
        enable_reward_learning=True, da_gain=da_gain,
        noise_std=1.8, noise_sqrt_dt=True,
        seek_interval=10**9, prune_interval=10**9,          # freeze structure
        neuromod_lr=0.02, tau_elig=300.0, tau_dopamine=200.0, w_max=2.0,
    )


def build(N: int, device, seed: int, da_gain: float = 1.0):
    """
    Build a network with two initially-identical, weak projections:
      • cue → action        the INCENTIVE pathway (driven by the cue, reward-paired)
      • ctrl → ctrl_out     a CONTROL pathway (never driven, never rewarded) targeting
                            a *separate, quiet* output group, so the directed-credit
                            marker measures reward specificity without post-group leakage.
    Returns (brain, cue, action, ctrl, ctrl_out).
    """
    brain = BrainNet(N=N, params=_params(da_gain), device=device, seed=seed)
    brain.synapses.W *= 2.0                              # responsive regime

    cue     = list(range(0, 20))
    ctrl    = list(range(20, 40))                        # never-driven control input
    action  = list(range(N - 30, N))                    # rewarded output
    ctrl_out = list(range(N - 60, N - 30))              # separate, unrewarded output

    # Make the pathway neurons excitatory so cue drive is unambiguous.
    brain.population.is_inhibitory[cue + ctrl + action + ctrl_out] = False

    # Seed both projections identically weak, so any later asymmetry is caused by
    # the (contingent) reward, not by initial conditions.
    W = brain.synapses.W
    for pre, post in ((cue, action), (ctrl, ctrl_out)):
        pi = torch.tensor(pre, device=W.device)
        po = torch.tensor(post, device=W.device)
        W[pi.unsqueeze(1), po.unsqueeze(0)] = 0.30
        brain.synapses.mask[pi.unsqueeze(1), po.unsqueeze(0)] = True
    return brain, cue, action, ctrl, ctrl_out


# ═════════════════════════════════════════════════════════════════
#  Primitive operations
# ═════════════════════════════════════════════════════════════════

def _present(brain, cue, action, dt, stim_amp=6.0, stim_steps=60, post_steps=40) -> float:
    """Drive the cue; return total action-group spikes over stim+post window
    (the network's cue-evoked 'pursuit')."""
    N = brain.N
    stim = torch.zeros(N, device=brain.device); stim[cue] = stim_amp
    count = 0.0
    for _ in range(stim_steps):
        sp = brain.step(stim, dt=dt)[0]
        count += float(sp[action].sum().item())
    for _ in range(post_steps):
        sp = brain.step(dt=dt)[0]
        count += float(sp[action].sum().item())
    return count


def _rest(brain, steps, dt):
    for _ in range(steps):
        brain.step(dt=dt)


def _probe(brain, cue, action, dt, n=6) -> float:
    """Mean cue-evoked pursuit over `n` presentations, WITHOUT reward. Dopamine
    sits at baseline so three-factor consolidation is inert — probing does not
    teach (verified: M = da_gain·(dopamine−1) = 0 at baseline)."""
    vals = [_present(brain, cue, action, dt) for _ in range(n)]
    return float(np.mean(vals))


def _path_mean(brain, pre, action) -> float:
    W = brain.synapses.W
    return float(W[pre][:, action].mean().item())


# ═════════════════════════════════════════════════════════════════
#  Protocol — one condition, run end to end
# ═════════════════════════════════════════════════════════════════

def run_condition(name: str, N: int, device, seed: int, dt: float = 1e-4,
                  n_train: int = 60, n_ext: int = 30, n_reinstate: int = 12,
                  reward_mag: float = 1.0, settle: int = 60, rest: int = 40,
                  verbose: bool = True) -> dict:
    """
    Phases: warm-up → NAÏVE probe → rewarded TRAINING → TRAINED probe →
            EXTINCTION (reward withdrawn) → EXTINGUISHED probe →
            REINSTATEMENT (reward restored) → REINSTATED probe.

    `name` selects the condition: 'INTACT', 'DA-LESION', or 'DECOUPLED'.
    """
    da_gain = 0.0 if name == 'DA-LESION' else 1.0
    decoupled = (name == 'DECOUPLED')
    brain, cue, action, ctrl, ctrl_out = build(N, device, seed, da_gain=da_gain)

    _rest(brain, 300, dt)                                            # warm-up

    w_cue0, w_ctrl0 = _path_mean(brain, cue, action), _path_mean(brain, ctrl, ctrl_out)
    naive = _probe(brain, cue, action, dt)
    # Operant criterion: reward when the cue evokes action at/above half the naive
    # rate. (The 6-probe naive MEAN sits above most single-trial values, so a full
    # naive threshold rewards too rarely to bootstrap; half-naive reliably shapes
    # up. The SAME threshold is used in every condition, matching the reward rate.)
    thr = 0.5 * max(naive, 1.0)

    def deliver_reward():
        brain.reward(1.0 + reward_mag)                              # dopamine burst
        _rest(brain, settle, dt)                                    # consolidate eligible

    def deliver_dip():
        # Omission of an expected reward → phasic dopamine dip → negative RPE →
        # LTD of the eligible pathway (extinction). Schultz 1997.
        brain.modulator.dopamine = max(0.0, 1.0 - reward_mag)
        _rest(brain, settle, dt)

    # ── TRAINING: reward contingent on cue-evoked action (operant) ──
    acq_curve = []
    for t in range(n_train):
        pursuit = _present(brain, cue, action, dt)
        acq_curve.append(pursuit)
        if decoupled:
            # Non-contingent (yoked) reward: SAME dopamine, but unpaired from the
            # cue→action pathway. Clearing the eligibility (credit) trace before the
            # reward instantiates "the reinforcer is not credited to the action just
            # performed" — the dopamine burst finds no tagged cue→action synapse to
            # potentiate. Reward *rate* is matched to INTACT (same threshold).
            _rest(brain, rest, dt)
            if pursuit >= thr:
                if brain.synapses.elig is not None:
                    brain.synapses.elig.zero_()
                deliver_reward()
        else:
            if pursuit >= thr:
                deliver_reward()
            else:
                _rest(brain, settle, dt)
            _rest(brain, rest, dt)

    trained = _probe(brain, cue, action, dt)
    w_cue1, w_ctrl1 = _path_mean(brain, cue, action), _path_mean(brain, ctrl, ctrl_out)

    # ── EXTINCTION: cue presented, reward withheld → negative RPE → LTD ──
    ext_curve = []
    for t in range(n_ext):
        pursuit = _present(brain, cue, action, dt)
        ext_curve.append(pursuit)
        deliver_dip()
        _rest(brain, rest, dt)
    extinguished = _probe(brain, cue, action, dt)

    # ── REINSTATEMENT: re-pair reward with the cue; measure recovery. Each
    #    condition keeps its own contingency rule (decoupled stays unpaired,
    #    lesion's dopamine still cannot gate) so recovery itself is diagnostic. ──
    rei_curve = []
    for t in range(n_reinstate):
        pursuit = _present(brain, cue, action, dt)
        rei_curve.append(pursuit)
        if decoupled and brain.synapses.elig is not None:
            brain.synapses.elig.zero_()                            # reward stays unpaired
        deliver_reward()                                           # unconditional priming
        _rest(brain, rest, dt)
    reinstated = _probe(brain, cue, action, dt)

    # ── Markers (all oriented so higher = more desire-like) ──
    salience_gain   = trained - naive                              # 1. acquisition of wanting
    pursuit_asympt  = trained                                      # 2. wanting magnitude
    reinstatement   = reinstated - extinguished                    # 3. revivable incentive
    directed_credit = (w_cue1 - w_cue0) - (w_ctrl1 - w_ctrl0)      # 4. object-directedness

    res = dict(
        condition=name,
        naive=naive, trained=trained, extinguished=extinguished, reinstated=reinstated,
        salience_gain=salience_gain, pursuit_asymptote=pursuit_asympt,
        reinstatement=reinstatement, directed_credit=directed_credit,
        w_cue=(w_cue0, w_cue1), w_ctrl=(w_ctrl0, w_ctrl1),
        acq_curve=acq_curve, ext_curve=ext_curve, rei_curve=rei_curve,
    )
    if verbose:
        print(f"  [{name:9s}] naive={naive:6.1f}  trained={trained:6.1f}  "
              f"extinq={extinguished:6.1f}  reinstated={reinstated:6.1f}  "
              f"| dW_cue={w_cue1-w_cue0:+.3f} dW_ctrl={w_ctrl1-w_ctrl0:+.3f}")
    return res


# ═════════════════════════════════════════════════════════════════
#  Battery runner + aggregate desire index
# ═════════════════════════════════════════════════════════════════

CONDITIONS = ['INTACT', 'DA-LESION', 'DECOUPLED']
MARKERS = ['salience_gain', 'pursuit_asymptote', 'reinstatement', 'directed_credit']
MARKER_LABELS = {
    'salience_gain'    : 'incentive salience (acquired wanting)',
    'pursuit_asymptote': 'pursuit asymptote (wanting magnitude)',
    'reinstatement'    : 'reinstatement (revivable incentive)',
    'directed_credit'  : 'directed credit (object-directedness)',
}


def run_battery(N: int = 160, device=None, seed: int = 0, verbose: bool = True) -> dict:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"\n{'='*72}\n DESIRE-MARKER BATTERY  (N={N}, device={device}, seed={seed})\n{'='*72}")
        print(" Measures operational markers of incentive motivation ('wanting') and")
        print(" validates them by dissociation from dopamine-lesioned and reward-")
        print(" decoupled controls. It does NOT prove the network wants anything.\n")

    rows = {c: run_condition(c, N, device, seed, verbose=verbose) for c in CONDITIONS}

    # Aggregate: z-score each marker across conditions, average → desire index.
    M = np.array([[rows[c][m] for m in MARKERS] for c in CONDITIONS], dtype=float)
    mu, sd = M.mean(0), M.std(0) + 1e-9
    Z = (M - mu) / sd
    desire = Z.mean(1)
    for i, c in enumerate(CONDITIONS):
        rows[c]['desire_index'] = float(desire[i])

    if verbose:
        _report(rows)
    return rows


def _report(rows: dict):
    print(f"\n{'-'*86}")
    print(f"{'marker':<42}" + "".join(f"{c:>15}" for c in CONDITIONS))
    print(f"{'-'*86}")
    for m in MARKERS:
        print(f"{MARKER_LABELS[m]:<42}" + "".join(f"{rows[c][m]:>15.3f}" for c in CONDITIONS))
    print(f"{'-'*86}")
    print(f"{'DESIRE INDEX (z-avg)':<42}" + "".join(f"{rows[c]['desire_index']:>15.3f}" for c in CONDITIONS))
    print(f"{'-'*86}")
    best = max(rows, key=lambda c: rows[c]['desire_index'])
    print(f"\n  Top scorer: {best}")
    if best == 'INTACT' and all(rows['INTACT']['desire_index'] > rows[c]['desire_index']
                                for c in rows if c != 'INTACT'):
        print("  [PASS] The intact, dopamine-competent, contingently-trained network")
        print("    dissociates from both controls -- the expected incentive-motivation pattern.")
    else:
        print("  [FAIL] No clean dissociation -- retune (reward_mag, N, n_train) and retry.")
    print("\n  Interpretation: evidence of desire-like DYNAMICS (dopamine-gated, cue-")
    print("  directed, reward-contingent pursuit), not proof of subjective wanting.")


if __name__ == '__main__':
    import sys, json
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    out = run_battery(N=N)
    # Persist a compact JSON for the report generator.
    slim = {c: {k: v for k, v in out[c].items()
                if k in MARKERS + ['desire_index', 'naive', 'trained', 'extinguished',
                                   'reinstated', 'w_cue', 'w_ctrl', 'acq_curve',
                                   'ext_curve', 'rei_curve']}
            for c in out}
    with open('desire_results.json', 'w') as f:
        json.dump(slim, f)
    print("\n  wrote desire_results.json")
