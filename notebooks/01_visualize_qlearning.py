"""
01_visualize_qlearning.py
Visualize Q-Learning results — 4 plots:
  A) Per-env: 4 decay types (training curve + success rate + epsilon)
  B) Per-decay: 6 envs (training curve + success rate + steps)
  C) Overview heatmap (success rate & avg reward)
  D) Cross-env: Empty group & DoorKey group
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *

ALGO   = 'qlearning'
LABEL  = 'Q-Learning'
COLOR  = 'royalblue'
OUT    = os.path.join(PLOTS_DIR, 'qlearning')
os.makedirs(OUT, exist_ok=True)

# ── A) Per-env: 4 decays so sánh ─────────────────────────────────────────────
print("[A] Per-env plots...")
for env_id, env_label in ENVS:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{LABEL} — {env_label}', fontsize=13, fontweight='bold')
    ax_r, ax_sr, ax_eps = axes

    for decay in DECAYS:
        st = load(ALGO, env_id, decay)
        if st is None: continue
        c, ls, lbl = DECAY_COLORS[decay], DECAY_STYLES[decay], DECAY_LABELS[decay]
        sm = summary(st)
        ax_r.plot(smooth(st['rewards']),  color=c, ls=ls, lw=1.8, label=lbl)
        ax_sr.plot(smooth(st['success']), color=c, ls=ls, lw=1.8, label=lbl)
        ax_eps.plot(st['epsilons'],        color=c, ls=ls, lw=1.2, alpha=0.9, label=lbl)
        # annotate first success
        if sm and sm['first']:
            ax_sr.axvline(sm['first'], color=c, ls=':', lw=0.8, alpha=0.5)

    ax_r.axhline(0, color='k', ls='--', lw=0.6, alpha=0.4)
    for ax, ylabel, title in [
        (ax_r,   'Total Reward (avg 100)',  'Training Curve'),
        (ax_sr,  'Success Rate (avg 100)',  'Success Rate'),
        (ax_eps, 'Epsilon ε',               'Exploration Rate'),
    ]:
        ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax_sr.set_ylim(-0.05, 1.05)
    ax_eps.set_ylim(0, 1.05)

    plt.tight_layout()
    savefig(fig, f'{OUT}/{ALGO}__{env_id}__decay_compare.png')

# ── B) Per-decay: 6 envs so sánh ─────────────────────────────────────────────
print("[B] Per-decay plots...")
ENV_COLORS = {
    'empty_5x5':'#1f77b4','empty_8x8':'#4aa3df','empty_16x16':'#aec7e8',
    'doorkey_5x5':'#d62728','doorkey_8x8':'#e07055','doorkey_16x16':'#f5b8b0',
}
ENV_STYLES = {
    'empty_5x5':'-','empty_8x8':'--','empty_16x16':':',
    'doorkey_5x5':'-','doorkey_8x8':'--','doorkey_16x16':':',
}
for decay in DECAYS:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{LABEL} — {DECAY_LABELS[decay]}  |  All Environments',
                 fontsize=13, fontweight='bold')
    ax_r, ax_sr, ax_st = axes
    for env_id, env_label in ENVS:
        st = load(ALGO, env_id, decay)
        if st is None: continue
        c, ls = ENV_COLORS[env_id], ENV_STYLES[env_id]
        ax_r.plot(smooth(st['rewards']),  color=c, ls=ls, lw=1.8, label=env_label)
        ax_sr.plot(smooth(st['success']), color=c, ls=ls, lw=1.8, label=env_label)
        ax_st.plot(smooth(st['steps']),   color=c, ls=ls, lw=1.4, label=env_label)
    ax_r.axhline(0, color='k', ls='--', lw=0.6, alpha=0.4)
    # Divider Empty vs DoorKey
    for ax in [ax_r, ax_sr, ax_st]:
        ax.set_xlabel('Episode'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax_r.set_title('Training Curve');  ax_r.set_ylabel('Total Reward (avg 100)')
    ax_sr.set_title('Success Rate');   ax_sr.set_ylabel('Success Rate (avg 100)'); ax_sr.set_ylim(-0.05,1.05)
    ax_st.set_title('Steps/Episode');  ax_st.set_ylabel('Steps (avg 100)')
    plt.tight_layout()
    savefig(fig, f'{OUT}/{ALGO}__{decay}__env_compare.png')

# ── C) Heatmap ────────────────────────────────────────────────────────────────
print("[C] Heatmap...")
env_labels   = [e[1] for e in ENVS]
decay_labels = [DECAY_LABELS[d] for d in DECAYS]
sr_mat  = np.full((len(DECAYS), len(ENVS)), np.nan)
r_mat   = np.full((len(DECAYS), len(ENVS)), np.nan)
for i, decay in enumerate(DECAYS):
    for j, (env_id, _) in enumerate(ENVS):
        sm = summary(load(ALGO, env_id, decay))
        if sm: sr_mat[i,j] = sm['sr']; r_mat[i,j] = sm['avg_r']

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
fig.suptitle(f'{LABEL} — Summary Heatmap (last 500 eps)', fontsize=13, fontweight='bold')
for ax, mat, title, fmt, cmap in [
    (axes[0], sr_mat, 'Success Rate', 'sr',  'RdYlGn'),
    (axes[1], r_mat,  'Avg Reward',   'r',   'RdYlGn'),
]:
    vmin = np.nanmin(mat); vmax = np.nanmax(mat)
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ENVS)));   ax.set_xticklabels(env_labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(DECAYS))); ax.set_yticklabels(decay_labels, fontsize=9)
    ax.axvline(2.5, color='white', lw=2.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for i in range(len(DECAYS)):
        for j in range(len(ENVS)):
            v = mat[i,j]
            if np.isnan(v): continue
            txt = f'{v:.0%}' if fmt=='sr' else f'{v:.2f}'
            nv  = (v-vmin)/(vmax-vmin+1e-9)
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white' if nv<0.35 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
plt.tight_layout()
savefig(fig, f'{OUT}/{ALGO}__heatmap.png')

# ── D) Cross-env: Empty vs DoorKey ───────────────────────────────────────────
print("[D] Cross-env...")
for decay in DECAYS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{LABEL} — {DECAY_LABELS[decay]}  |  Scaling with Environment Size',
                 fontsize=12, fontweight='bold')
    for col, (env_group, group_label) in enumerate([(ENVS_EMPTY,'Empty'),(ENVS_DOORKEY,'DoorKey')]):
        ax = axes[col]
        styles = ['-','--',':']
        alphas = [1.0, 0.7, 0.45]
        for (env_id, env_label), ls, alpha in zip(env_group, styles, alphas):
            st = load(ALGO, env_id, decay)
            if st is None: continue
            ax.plot(smooth(st['success']), color=COLOR, ls=ls, alpha=alpha, lw=2, label=env_label)
        ax.set_title(f'{group_label} Environments'); ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (avg 100)'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    savefig(fig, f'{OUT}/{ALGO}__{decay}__cross_env.png')

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"{'Env':<16} {'Decay':<20} {'SR(500)':>8} {'SR(100)':>8} {'AvgR':>8} {'First':>8} {'PeakSR':>8}")
print(f"{'-'*80}")
for env_id, env_label in ENVS:
    for decay in DECAYS:
        sm = summary(load(ALGO, env_id, decay))
        if not sm: continue
        fe = f"ep{sm['first']}" if sm['first'] else 'Never'
        print(f"  {env_label:<14} {DECAY_LABELS[decay]:<20} {sm['sr']:>7.1%} "
              f"{sm['sr100']:>7.1%} {sm['avg_r']:>8.3f} {fe:>8} {sm['peak_sr']:>7.1%}")
    print()

print(f"\nDone! All Q-Learning plots saved to {OUT}/")
