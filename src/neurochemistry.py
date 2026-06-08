"""
neurochemistry.py — Extensible neurochemical system for BrainNet
================================================================
A composable, biologically-grounded neurochemistry layer that plugs into
`core.py` without changing it. The trick: the simulation responds to five
"control axes" (dopamine, serotonin, norepinephrine, acetylcholine, cortisol),
and **every neurochemical exerts its influence by pushing those axes**. A
`Neurochemistry` object therefore behaves as a drop-in superset of
`NeuromodulatorState` — `brain.modulator.dopamine` etc. still work — while
tracking an arbitrary, user-extensible set of chemicals, each with its own
release/clearance dynamics.

    from core import BrainNet
    from neurochemistry import default_neurochemistry

    brain = BrainNet(N=500, neurochem=default_neurochemistry())
    brain.modulator.release('adenosine', 1.0)     # build up sleep pressure
    brain.modulator.release('cortisol', 1.5)      # acute stress
    brain.step()                                  # chemicals evolve each step

Add your own:

    from neurochemistry import Neurochemical
    brain.modulator.add(Neurochemical(
        'caffeine', baseline=0.0, tau=20000.0, kind='xenobiotic',
        effects={'norepinephrine': +0.4, 'acetylcholine': +0.2},   # adenosine antagonist
        description='blocks sleep pressure → net arousal'))

Designed as a library primitive: the framework is generic, the default set is a
credible starting palette, and powerful users can register custom chemicals,
receptors (axis effects), and dynamics for larger-scale experiments.

References: Schultz 1997 (dopamine); Berridge & Robinson 1998; Joëls & Krugers
2007 (cortisol); Porkka-Heiskanen 1997 (adenosine/sleep); Saper et al. 2005
(sleep-wake neurochemistry).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

__all__ = ['Neurochemical', 'Neurochemistry', 'default_neurochemistry',
           'AXIS_BASELINE', 'stressed', 'drowsy', 'aroused']

# The control axes the core simulation reads, with their resting values.
# (dopamine→LTP gain, serotonin→tau_m, norepinephrine→noise/threshold,
#  acetylcholine→synapse-seeking, cortisol→stress/LTD.) Every chemical's
# influence is expressed as a push on one or more of these.
AXIS_BASELINE: Dict[str, float] = {
    'dopamine': 1.0, 'serotonin': 1.0, 'norepinephrine': 1.0,
    'acetylcholine': 1.0, 'cortisol': 0.0,
}


@dataclass
class Neurochemical:
    """
    One chemical species with first-order release/clearance kinetics.

    level relaxes exponentially toward `baseline` with time constant `tau` (ms);
    `release()` adds a bolus. `effects` maps control-axis name → sensitivity:
    the axis gains `coeff · (level − baseline)`, so a chemical at baseline has no
    effect and departures from baseline push the network.
    """
    name: str
    baseline: float = 0.0
    tau: float = 1000.0                                   # clearance time constant (ms)
    kind: str = 'neuromodulator'                          # neurotransmitter|neuromodulator|hormone|neuropeptide|gas|...
    effects: Dict[str, float] = field(default_factory=dict)   # axis -> sensitivity
    description: str = ''
    level: Optional[float] = None                         # current concentration

    def __post_init__(self):
        if self.level is None:
            self.level = self.baseline

    def relax(self, dt_ms: float):
        self.level = self.baseline + (self.level - self.baseline) * float(np.exp(-dt_ms / self.tau))

    def release(self, amount: float):
        self.level += amount

    def set(self, value: float):
        self.level = value


class Neurochemistry:
    """
    A bag of `Neurochemical`s that presents itself to `core.py` as a
    `NeuromodulatorState`. Reading an axis name (e.g. `.dopamine`) returns the
    *net* axis value summed over all chemicals; assigning to a chemical name
    (e.g. `.dopamine = 2.0`) sets that chemical's level. `step(dt)` advances all
    chemical kinetics.
    """

    def __init__(self, chemicals: Optional[List[Neurochemical]] = None,
                 axis_baseline: Optional[Dict[str, float]] = None):
        object.__setattr__(self, '_chem', {})
        object.__setattr__(self, '_axis_base', dict(axis_baseline or AXIS_BASELINE))
        for c in (chemicals or []):
            self._chem[c.name] = c

    # ── dynamics ──────────────────────────────────────────────────
    def step(self, dt: float):
        """Advance all chemical kinetics by dt seconds (clearance toward baseline)."""
        dt_ms = dt * 1000.0
        for c in self._chem.values():
            c.relax(dt_ms)

    def axis(self, name: str) -> float:
        """Net value of a control axis = baseline + Σ chemical pushes."""
        total = self._axis_base.get(name, 0.0)
        for c in self._chem.values():
            coeff = c.effects.get(name)
            if coeff:
                total += coeff * (c.level - c.baseline)
        return total

    # ── NeuromodulatorState-compatible attribute access ───────────
    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails.
        axis_base = self.__dict__.get('_axis_base', {})
        chem = self.__dict__.get('_chem', {})
        if name in axis_base:
            return self.axis(name)          # network-facing axis value
        if name in chem:
            return chem[name].level         # raw chemical level
        raise AttributeError(f"Neurochemistry has no axis or chemical '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        chem = self.__dict__.get('_chem', {})
        if name in chem:
            chem[name].level = value        # e.g. brain.modulator.dopamine = 2.0
        else:
            object.__setattr__(self, name, value)

    # ── library API ───────────────────────────────────────────────
    def release(self, name: str, amount: float):
        self._chem[name].release(amount)

    def set_level(self, name: str, value: float):
        self._chem[name].level = value

    def get(self, name: str) -> Neurochemical:
        return self._chem[name]

    def add(self, chem: Neurochemical):
        """Register a new chemical (or replace one by name)."""
        self._chem[chem.name] = chem

    def register_axis(self, name: str, baseline: float = 0.0):
        """Add a custom control axis (for users wiring new effects into a fork)."""
        self._axis_base[name] = baseline

    def names(self) -> List[str]:
        return list(self._chem)

    def levels(self) -> Dict[str, float]:
        return {n: c.level for n, c in self._chem.items()}

    def axes(self) -> Dict[str, float]:
        return {a: self.axis(a) for a in self._axis_base}

    def snapshot(self) -> dict:
        return {'levels': self.levels(), 'axes': self.axes()}

    def __repr__(self):
        ax = ', '.join(f"{k}={v:.2f}" for k, v in self.axes().items())
        return f"Neurochemistry({len(self._chem)} chemicals | axes: {ax})"


# ═════════════════════════════════════════════════════════════════
#  Default palette — a credible starting neurochemistry
# ═════════════════════════════════════════════════════════════════

def default_neurochemistry() -> Neurochemistry:
    """
    A biologically-grounded default set. The five *primary* species map 1:1 onto
    their control axis (so with only these active the network behaves exactly as
    with NeuromodulatorState). The *secondary* species (histamine, adenosine,
    melatonin, oxytocin, endorphin) push those same axes — modelling wake/sleep
    and stress-buffering chemistry. Fast transmitters and gases are included as
    first-class, dynamically-tracked species for completeness and extension.
    """
    return Neurochemistry([
        # ── Primary modulators (1:1 with their axis) ──
        Neurochemical('dopamine',       1.0,  200.0, 'neuromodulator',
                      {'dopamine': 1.0},        'reward / reinforcement (VTA, SNc)'),
        Neurochemical('serotonin',      1.0, 3000.0, 'neuromodulator',
                      {'serotonin': 1.0},       'mood / stability (raphe nuclei)'),
        Neurochemical('norepinephrine', 1.0,  500.0, 'neuromodulator',
                      {'norepinephrine': 1.0},  'arousal / vigilance (locus coeruleus)'),
        Neurochemical('acetylcholine',  1.0,  300.0, 'neuromodulator',
                      {'acetylcholine': 1.0},   'attention / plasticity (basal forebrain)'),
        Neurochemical('cortisol',       0.0, 3000.0, 'hormone',
                      {'cortisol': 1.0},        'stress glucocorticoid (HPA axis)'),
        # ── Secondary modulators that push the canonical axes ──
        Neurochemical('histamine',      0.0, 1000.0, 'neuromodulator',
                      {'norepinephrine': 0.30, 'acetylcholine': 0.10},
                      'wakefulness (tuberomammillary nucleus)'),
        Neurochemical('adenosine',      0.0, 8000.0, 'neuromodulator',
                      {'norepinephrine': -0.40, 'acetylcholine': -0.30},
                      'sleep pressure; accumulates with waking, blocked by caffeine'),
        Neurochemical('melatonin',      0.0, 8000.0, 'hormone',
                      {'norepinephrine': -0.30, 'serotonin': 0.20},
                      'circadian sleep signal (pineal gland)'),
        Neurochemical('oxytocin',       0.0, 4000.0, 'neuropeptide',
                      {'cortisol': -0.30, 'dopamine': 0.10},
                      'social bonding; buffers the stress response'),
        Neurochemical('endorphin',      0.0, 2000.0, 'neuropeptide',
                      {'cortisol': -0.40, 'dopamine': 0.20},
                      'analgesia / reward; stress relief'),
        # ── Fast transmitters & gases (tracked; effects left to user wiring) ──
        Neurochemical('glutamate',      1.0,   10.0, 'neurotransmitter', {},
                      'principal excitatory transmitter (global tone)'),
        Neurochemical('GABA',           1.0,   10.0, 'neurotransmitter', {},
                      'principal inhibitory transmitter (global tone)'),
        Neurochemical('nitric_oxide',   0.0,  200.0, 'gas', {},
                      'diffusible retrograde messenger'),
        Neurochemical('endocannabinoid',0.0, 1000.0, 'lipid', {},
                      'retrograde signal; suppresses transmitter release'),
    ])


# ── Convenience scenario presets ──────────────────────────────────

def stressed(level: float = 1.5) -> Neurochemistry:
    """Acute stress: elevated cortisol."""
    nc = default_neurochemistry(); nc.set_level('cortisol', level); return nc


def drowsy(adenosine: float = 1.0, melatonin: float = 0.8) -> Neurochemistry:
    """High sleep pressure: adenosine + melatonin suppress arousal/attention."""
    nc = default_neurochemistry()
    nc.release('adenosine', adenosine); nc.release('melatonin', melatonin)
    return nc


def aroused(ne: float = 0.5, histamine: float = 0.6) -> Neurochemistry:
    """Heightened arousal: extra norepinephrine + histamine."""
    nc = default_neurochemistry()
    nc.release('norepinephrine', ne); nc.release('histamine', histamine)
    return nc
