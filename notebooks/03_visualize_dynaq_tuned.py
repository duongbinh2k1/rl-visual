"""
03_visualize_dynaq_tuned.py — Dyna-Q (n=5) visualization
Same plot structure as Q-Learning for consistent comparison
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *

ALGO  = 'dynaq_tuned'
LABEL = 'Dyna-Q Tuned (n=20, lr=0.2)'
COLOR = 'teal'
OUT   = os.path.join(PLOTS_DIR, 'dynaq_tuned')
os.makedirs(OUT, exist_ok=True)

ENV_COLORS = {
    'empty_5x5':'#1f77b4','empty_8x8':'#4aa3df','empty_16x16':'#aec7e8',
    'doorkey_5x5':'#d62728','doorkey_8x8':'#e07055','doorkey_16x16':'#f5b8b0',
}
ENV_STYLES = {'empty_5x5':'-','empty_8x8':'--','empty_16x16':':',
              'doorkey_5x5':'-','doorkey_8x8':'--','doorkey_16x16':':'}

# A) Per-env
print("[A] Per-env..."); 
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
        if sm and sm['first']:
            ax_sr.axvline(sm['first'], color=c, ls=':', lw=0.8, alpha=0.5)
    ax_r.axhline(0, color='k', ls='--', lw=0.6, alpha=0.4)
    for ax, ylabel, title in [(ax_r,'Total Reward (avg 100)','Training Curve'),
                               (ax_sr,'Success Rate (avg 100)','Success Rate'),
                               (ax_eps,'Epsilon ε','Exploration Rate')]:
        ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax_sr.set_ylim(-0.05, 1.05); ax_eps.set_ylim(0, 1.05)
    plt.tight_layout()
    savefig(fig, f'{OUT}/{ALGO}__{env_id}__decay_compare.png')

# B) Per-decay
print("[B] Per-decay...")
for decay in DECAYS:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{LABEL} — {DECAY_LABELS[decay]}', fontsize=13, fontweight='bold')
    ax_r, ax_sr, ax_st = axes
    for env_id, env_label in ENVS:
        st = load(ALGO, env_id, decay)
        if st is None: continue
        c, ls = ENV_COLORS[env_id], ENV_STYLES[env_id]
        ax_r.plot(smooth(st['rewards']),  color=c, ls=ls, lw=1.8, label=env_label)
        ax_sr.plot(smooth(st['success']), color=c, ls=ls, lw=1.8, label=env_label)
        ax_st.plot(smooth(st['steps']),   color=c, ls=ls, lw=1.4, label=env_label)
    ax_r.axhline(0, color='k', ls='--', lw=0.6, alpha=0.4)
    for ax in [ax_r, ax_sr, ax_st]: ax.set_xlabel('Episode'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax_r.set_title('Training Curve');  ax_r.set_ylabel('Total Reward (avg 100)')
    ax_sr.set_title('Success Rate');   ax_sr.set_ylabel('Success Rate (avg 100)'); ax_sr.set_ylim(-0.05,1.05)
    ax_st.set_title('Steps/Episode');  ax_st.set_ylabel('Steps (avg 100)')
    plt.tight_layout()
    savefig(fig, f'{OUT}/{ALGO}__{decay}__env_compare.png')

# C) Heatmap
print("[C] Heatmap...")
sr_mat = np.full((len(DECAYS), len(ENVS)), np.nan)
r_mat  = np.full((len(DECAYS), len(ENVS)), np.nan)
for i,decay in enumerate(DECAYS):
    for j,(env_id,_) in enumerate(ENVS):
        sm = summary(load(ALGO, env_id, decay))
        if sm: sr_mat[i,j]=sm['sr']; r_mat[i,j]=sm['avg_r']
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
fig.suptitle(f'{LABEL} — Summary Heatmap', fontsize=13, fontweight='bold')
for ax, mat, title, fmt in [(axes[0],sr_mat,'Success Rate','sr'),(axes[1],r_mat,'Avg Reward','r')]:
    vmin=np.nanmin(mat); vmax=np.nanmax(mat)
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ENVS))); ax.set_xticklabels([e[1] for e in ENVS], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(DECAYS))); ax.set_yticklabels([DECAY_LABELS[d] for d in DECAYS], fontsize=9)
    ax.axvline(2.5, color='white', lw=2.5); ax.set_title(title, fontsize=11, fontweight='bold')
    for i in range(len(DECAYS)):
        for j in range(len(ENVS)):
            v=mat[i,j]
            if np.isnan(v): continue
            txt=f'{v:.0%}' if fmt=='sr' else f'{v:.2f}'
            nv=(v-vmin)/(vmax-vmin+1e-9)
            ax.text(j,i,txt,ha='center',va='center',fontsize=8,fontweight='bold',color='white' if nv<0.35 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
plt.tight_layout()
savefig(fig, f'{OUT}/{ALGO}__heatmap.png')

# D) avg_q progression (Dyna-Q specific — shows how Q-table fills up)
print("[D] avg_q progression...")
for env_id, env_label in ENVS:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f'{LABEL} — avg |Q-value| Progression — {env_label}', fontweight='bold')
    for decay in DECAYS:
        st = load(ALGO, env_id, decay)
        if st is None: continue
        ax.plot(smooth(st['avg_q'], 50), color=DECAY_COLORS[decay],
                ls=DECAY_STYLES[decay], lw=1.6, label=DECAY_LABELS[decay])
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean |Q-value|')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, f'{OUT}/{ALGO}__{env_id}__avg_q.png')

print(f"\nDone! All Dyna-Q plots saved to {OUT}/")
