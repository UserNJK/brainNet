"""
run_desire_study.py — multi-seed run of the desire battery + publication figures.
Writes docs/desire_study.json (aggregated) and docs/desire_figs/*.png.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import desire_battery as db

SEEDS = [0, 1, 2, 3, 4]
N = 160
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, '..', 'docs'))
FIGS = os.path.join(DOCS, 'desire_figs')
os.makedirs(FIGS, exist_ok=True)

# Colorblind-safe categorical palette (Okabe–Ito; validated ΔE>12), fixed order.
COND_COLOR = {'INTACT': '#0072B2', 'DA-LESION': '#E69F00', 'DECOUPLED': '#009E73'}
INK, MUTED, GRID = '#1a1a2e', '#555555', '#d8d8d8'
PHASES = ['naive', 'trained', 'extinguished', 'reinstated']


def _style(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MUTED); ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


# ── run every seed ────────────────────────────────────────────────
def run():
    dev = torch.device('cpu')                        # small N: CPU beats GPU sync
    t0 = time.time()
    per = {c: {'markers': {m: [] for m in db.MARKERS}, 'desire_index': [],
               'phases': {p: [] for p in PHASES},
               'acq': [], 'ext': [], 'rei': [],
               'dW_cue': [], 'dW_ctrl': []}
           for c in db.CONDITIONS}
    for s in SEEDS:
        print(f"\n===== SEED {s} =====")
        rows = db.run_battery(N=N, device=dev, seed=s, verbose=True)
        for c in db.CONDITIONS:
            r = rows[c]
            for m in db.MARKERS:
                per[c]['markers'][m].append(r[m])
            per[c]['desire_index'].append(r['desire_index'])
            for p in PHASES:
                per[c]['phases'][p].append(r[p])
            per[c]['acq'].append(r['acq_curve'])
            per[c]['ext'].append(r['ext_curve'])
            per[c]['rei'].append(r['rei_curve'])
            per[c]['dW_cue'].append(r['w_cue'][1] - r['w_cue'][0])
            per[c]['dW_ctrl'].append(r['w_ctrl'][1] - r['w_ctrl'][0])
    print(f"\nTOTAL {time.time()-t0:.1f}s over {len(SEEDS)} seeds")

    # ── aggregate mean/sd ─────────────────────────────────────────
    def ms(x): x = np.asarray(x, float); return float(x.mean()), float(x.std())
    agg = {}
    for c in db.CONDITIONS:
        agg[c] = {
            'markers': {m: ms(per[c]['markers'][m]) for m in db.MARKERS},
            'desire_index': ms(per[c]['desire_index']),
            'phases': {p: ms(per[c]['phases'][p]) for p in PHASES},
            'dW_cue': ms(per[c]['dW_cue']),
            'dW_ctrl': ms(per[c]['dW_ctrl']),
            'acq_mean': np.mean(per[c]['acq'], axis=0).tolist(),
            'ext_mean': np.mean(per[c]['ext'], axis=0).tolist(),
            'rei_mean': np.mean(per[c]['rei'], axis=0).tolist(),
        }
    out = {'seeds': SEEDS, 'N': N, 'conditions': db.CONDITIONS,
           'markers': db.MARKERS, 'marker_labels': db.MARKER_LABELS,
           'agg': agg}
    with open(os.path.join(DOCS, 'desire_study.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print("wrote desire_study.json")
    make_figures(per, agg)
    return out


# ── figures ───────────────────────────────────────────────────────
def make_figures(per, agg):
    plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans',
                         'axes.titlecolor': INK, 'axes.labelcolor': INK,
                         'text.color': INK, 'figure.dpi': 150})

    # Fig 1 — acquisition curves (mean over seeds), the core "wanting is learned"
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for c in db.CONDITIONS:
        y = agg[c]['acq_mean']; x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=COND_COLOR[c], lw=2.0, label=c)
        ax.text(x[-1] + 0.6, y[-1], c, color=COND_COLOR[c], fontsize=8.5, va='center')
    ax.set_xlabel('rewarded training trial'); ax.set_ylabel('cue-evoked pursuit\n(action spikes / trial)')
    ax.set_title('Acquisition of incentive salience', fontweight='bold', loc='left', fontsize=11)
    ax.set_xlim(1, len(y) + 12); _style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'fig1_acquisition.png'), bbox_inches='tight'); plt.close(fig)

    # Fig 2 — phase timeline naive→trained→extinguished→reinstated (mean±sd)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xp = np.arange(len(PHASES))
    for c in db.CONDITIONS:
        m = np.array([agg[c]['phases'][p][0] for p in PHASES])
        sd = np.array([agg[c]['phases'][p][1] for p in PHASES])
        ax.plot(xp, m, '-o', color=COND_COLOR[c], lw=2.0, ms=6, label=c)
        ax.fill_between(xp, m - sd, m + sd, color=COND_COLOR[c], alpha=0.13, lw=0)
    ax.set_xticks(xp); ax.set_xticklabels(['naïve', 'trained', 'extinguished', 'reinstated'])
    ax.set_ylabel('cue-evoked pursuit'); ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.set_title('Desire life-cycle: acquire → extinguish → reinstate',
                 fontweight='bold', loc='left', fontsize=11)
    _style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'fig2_timeline.png'), bbox_inches='tight'); plt.close(fig)

    # Fig 3 — the four markers as 2x2 small multiples (mean±sd bars)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))
    for ax, mk in zip(axes.ravel(), db.MARKERS):
        vals = [agg[c]['markers'][mk][0] for c in db.CONDITIONS]
        errs = [agg[c]['markers'][mk][1] for c in db.CONDITIONS]
        cols = [COND_COLOR[c] for c in db.CONDITIONS]
        bars = ax.bar(range(3), vals, yerr=errs, color=cols, edgecolor=INK, lw=0.6,
                      width=0.66, capsize=3, error_kw=dict(lw=0.9, ecolor=MUTED))
        ax.set_xticks(range(3)); ax.set_xticklabels(db.CONDITIONS, fontsize=7.5, rotation=8)
        ax.set_title(db.MARKER_LABELS[mk], fontsize=8.7, loc='left')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (max(vals) * 0.02 if max(vals) > 0 else 0.01),
                    f'{v:.2f}' if max(vals) < 5 else f'{v:.0f}', ha='center', va='bottom',
                    fontsize=7.5, color=INK)
        _style(ax); ax.margins(y=0.18)
    fig.suptitle('Desire markers by condition (mean ± s.d., 5 seeds)',
                 fontweight='bold', x=0.02, ha='left', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGS, 'fig3_markers.png'), bbox_inches='tight'); plt.close(fig)

    # Fig 4 — aggregate desire index (mean±sd)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    vals = [agg[c]['desire_index'][0] for c in db.CONDITIONS]
    errs = [agg[c]['desire_index'][1] for c in db.CONDITIONS]
    cols = [COND_COLOR[c] for c in db.CONDITIONS]
    bars = ax.bar(range(3), vals, yerr=errs, color=cols, edgecolor=INK, lw=0.6,
                  width=0.62, capsize=3, error_kw=dict(lw=0.9, ecolor=MUTED))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(db.CONDITIONS, fontsize=8.5)
    ax.set_ylabel('desire index  (z-avg of markers)')
    ax.set_title('Aggregate desire index', fontweight='bold', loc='left', fontsize=11)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.09),
                f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8.5, color=INK)
    _style(ax); ax.margins(y=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'fig4_index.png'), bbox_inches='tight'); plt.close(fig)

    # Fig 5 — directed credit: paired vs control pathway potentiation
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(3); w = 0.36
    cue_m = [agg[c]['dW_cue'][0] for c in db.CONDITIONS]
    cue_e = [agg[c]['dW_cue'][1] for c in db.CONDITIONS]
    ctl_m = [agg[c]['dW_ctrl'][0] for c in db.CONDITIONS]
    ctl_e = [agg[c]['dW_ctrl'][1] for c in db.CONDITIONS]
    ax.bar(x - w/2, cue_m, w, yerr=cue_e, color='#0b3d91', edgecolor=INK, lw=0.5,
           capsize=3, label='cue → action (paired)', error_kw=dict(lw=0.8, ecolor=MUTED))
    ax.bar(x + w/2, ctl_m, w, yerr=ctl_e, color='#b9c4de', edgecolor=INK, lw=0.5,
           capsize=3, label='ctrl → ctrl-out (unpaired)', error_kw=dict(lw=0.8, ecolor=MUTED))
    ax.set_xticks(x); ax.set_xticklabels(db.CONDITIONS, fontsize=8.5)
    ax.set_ylabel('Δ synaptic weight'); ax.legend(frameon=False, fontsize=8)
    ax.set_title('Directed credit vs global-dopamine leakage', fontweight='bold', loc='left', fontsize=11)
    _style(ax); ax.margins(y=0.18)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'fig5_credit.png'), bbox_inches='tight'); plt.close(fig)
    print(f"wrote 5 figures to {FIGS}")


if __name__ == '__main__':
    run()
