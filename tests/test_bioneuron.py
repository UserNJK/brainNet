"""
Fast regression tests for the BioNeuron library.

Run directly (`python tests/test_bioneuron.py`) or with pytest. These are the
sub-second correctness checks; the longer learning demos live in closed_loop.py
and are documented in RESULTS.md.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from core import (BrainNet, NeuronParams, NeuromodulatorState, SynapseMatrix)
from sparse_synapses import SparseSynapses
from neurochemistry import default_neurochemistry, Neurochemical

CPU = torch.device('cpu')


def test_stdp_causality():
    """Bi & Poo (1998): pre-before-post potentiates, post-before-pre depresses."""
    p, mod = NeuronParams(), NeuromodulatorState()
    def delta(order):
        s = SynapseMatrix(2, p, CPU); s.add_synapse(0, 1, 0.5)
        a, b = ([1., 0.], [0., 1.]) if order == 'pre_post' else ([0., 1.], [1., 0.])
        s.apply_stdp(torch.tensor(a), 1e-3, mod); s.apply_stdp(torch.tensor(b), 1e-3, mod)
        return s.W[0, 1].item() - 0.5
    assert delta('pre_post') > 0, "pre->post must potentiate (LTP)"
    assert delta('post_pre') < 0, "post->pre must depress (LTD)"


def test_sparse_matches_dense():
    """Sparse backend reproduces dense propagation and STDP exactly."""
    N, p = 120, NeuronParams()
    torch.manual_seed(0)
    pre = torch.randint(0, N, (2000,)); post = torch.randint(0, N, (2000,))
    lin = torch.unique(pre * N + post); pre, post = lin // N, lin % N
    keep = pre != post; pre, post = pre[keep], post[keep]
    w = torch.empty(pre.shape).uniform_(0.05, 0.5)
    dense = SynapseMatrix(N, p, CPU); dense.add_synapses_bulk(pre, post, w)
    sparse = SparseSynapses(N, p, CPU); sparse.add_synapses_bulk(pre, post, w)
    assert dense.num_synapses() == sparse.num_synapses()
    sf = torch.randn(N)
    assert torch.allclose(dense.propagate(sf), sparse.propagate(sf), atol=1e-4)
    tp, tq = torch.rand(N), torch.rand(N)
    for s in (dense, sparse): s.trace_pre = tp.clone(); s.trace_post = tq.clone()
    spk = (torch.rand(N) < 0.1).float()
    dense.apply_stdp(spk.clone(), 1e-3, NeuromodulatorState())
    sparse.apply_stdp(spk.clone(), 1e-3, NeuromodulatorState())
    assert torch.allclose(dense.W[sparse.pre, sparse.post], sparse.w, atol=1e-5)


def test_neurochemistry_axes():
    """Drop-in compatibility + secondary chemicals push the control axes."""
    nc = default_neurochemistry()
    assert abs(nc.dopamine - 1.0) < 1e-9 and abs(nc.cortisol) < 1e-9
    nc.dopamine = 2.0
    assert abs(nc.dopamine - 2.0) < 1e-9
    nc = default_neurochemistry(); ne0 = nc.norepinephrine
    nc.release('histamine', 1.0)
    assert nc.norepinephrine > ne0 + 0.25         # arousal up
    nc = default_neurochemistry(); nc.release('adenosine', 1.0)
    assert nc.norepinephrine < 0.95               # sleep pressure down
    # kinetics: a bolus clears toward baseline
    nc = default_neurochemistry(); nc.release('cortisol', 2.0)
    for _ in range(2000): nc.step(1e-4)
    assert nc.cortisol < 2.0


def test_adaptation_reduces_firing():
    """Spike-frequency adaptation lowers steady-state rate under constant drive."""
    def total(adapt):
        p = NeuronParams(noise_std=2.0, enable_adaptation=adapt, b_adapt=0.04)
        b = BrainNet(N=80, params=p, device=CPU, seed=3)
        I = torch.full((80,), 3.0)
        return sum(int(b.step(I, dt=1e-4)[0].sum()) for _ in range(800))
    assert total(True) <= total(False)


def test_reward_and_cortisol_learning():
    """Reward potentiates the active pathway; cortisol depresses it."""
    def run(signal):
        p = NeuronParams(enable_reward_learning=True, noise_std=2.0, noise_sqrt_dt=True,
                         seek_interval=10**9, prune_interval=10**9, neuromod_lr=0.03)
        b = BrainNet(N=120, params=p, device=CPU, seed=0)
        A = list(range(30))
        def gm():
            W, M = b.synapses.W, b.synapses.mask
            sub = M[A][:, A]
            return float(W[A][:, A][sub].mean()) if bool(sub.any()) else 0.0
        a0 = gm()
        for _ in range(6):
            for _ in range(80):
                I = torch.zeros(120); I[A] = 5.0; b.step(I, dt=1e-4)
            b.reward(2.0) if signal == 'reward' else b.punish(1.5)
            for _ in range(120): b.step(dt=1e-4)
        return a0, gm()
    a0, a1 = run('reward');   assert a1 > a0 + 0.01, "reward should potentiate"
    a0, a1 = run('punish');   assert a1 < a0 - 0.005, "cortisol should depress"


def test_adex_stable_and_adapting():
    """AdEx fires, stays numerically bounded, and adaptation reduces firing."""
    def total(b_adapt):
        p = NeuronParams(enable_exp=True, enable_adaptation=True, b_adapt=b_adapt,
                         noise_std=1.0, noise_sqrt_dt=True, seek_interval=10**9)
        b = BrainNet(N=60, params=p, device=CPU, seed=1)
        I = torch.full((60,), 4.0)
        tot = 0
        for _ in range(1200):
            sp = b.step(I, dt=1e-4)[0]; tot += int(sp.sum())
        assert not (torch.isnan(b.population.V).any() or torch.isinf(b.population.V).any())
        return tot
    assert total(0.0) > 0, "AdEx must fire"
    assert total(0.15) < total(0.0), "more adaptation -> fewer spikes"


def test_istdp_regulates_excitation():
    """Inhibitory STDP (Vogels 2011) suppresses runaway excitation over time."""
    def trace(istdp):
        p = NeuronParams(enable_istdp=istdp, eta_istdp=0.05, rho_target=4.0,
                         noise_std=2.0, noise_sqrt_dt=True,
                         seek_interval=10**9, prune_interval=10**9)
        b = BrainNet(N=140, params=p, device=CPU, seed=0); b.synapses.W *= 1.5
        I = torch.full((140,), 4.0)
        early = late = 0
        for t in range(2600):
            sp = b.step(I, dt=1e-4)[0]
            if t < 400: early += int(sp.sum())
            if t >= 2200: late += int(sp.sum())
        return early, late
    e_on, l_on = trace(True); _, l_off = trace(False)
    assert l_on < l_off and l_on < e_on, "iSTDP should rein in firing"


def test_dendritic_supralinear_summation():
    """Active dendrite sums supralinearly (dendritic-spike signature); a point
    neuron does not amplify the same distributed sub-threshold input."""
    K = 20
    def response(active, dendrite):
        p = NeuronParams(enable_dendrite=dendrite, enable_nmda=dendrite,
                         noise_std=0.15, noise_sqrt_dt=True,
                         seek_interval=10**9, prune_interval=10**9, A_plus=0.0, A_minus=0.0,
                         V_d_thresh=-50.0, dend_plateau=2.5, g_couple=0.7,
                         tau_dend_spike=40.0, tau_d=20.0)
        b = BrainNet(N=K + 1, params=p, device=CPU, seed=0)
        b.population.set_cell_params(is_inhibitory=torch.zeros(K + 1, dtype=torch.bool))
        b.synapses.clear()
        b.synapses.add_synapses_bulk(torch.arange(1, K + 1),
                                     torch.zeros(K, dtype=torch.long), torch.full((K,), 0.055))
        I = torch.zeros(K + 1); I[active] = 6.0
        return sum(int(b.step(I, dt=1e-4)[0][0].item()) for _ in range(600))
    A = list(range(1, 9)); B = list(range(9, 17))
    rA = response(A, True); rB = response(B, True); rAB = response(A + B, True)
    assert rAB > rA + rB, "active dendrite should sum supralinearly"
    assert response(A + B, False) < rAB, "point neuron should not amplify the same input"


def test_cell_types_fs_faster_than_rs():
    """Structured cortex: fast-spiking interneurons out-fire pyramidal cells."""
    from cortex import build_cortex, excitatory_indices, fast_spiking_indices
    b = build_cortex(N=300, columns=1, device=CPU, seed=1)
    exc = excitatory_indices(b); fs = fast_spiking_indices(b)
    assert len(fs) > 0 and len(exc) > 0
    I = torch.zeros(300); I[exc] = 3.0
    for _ in range(1200):
        b.step(I, dt=1e-4)
    fr = b.population.firing_rate
    assert float(fr[fs].mean()) > float(fr[exc].mean()), "FS should fire faster than RS"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    return failed


if __name__ == '__main__':
    raise SystemExit(1 if _run_all() else 0)
