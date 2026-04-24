"""
05_compare_all.py — Cross-algorithm comparison plots
All 4 algorithms compared side-by-side on same environment/decay
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *

OUT = os.path.join(PLOTS_DIR, 'comparison')
os.makedirs(OUT, exist_ok=True)

ALGO_LIST   = ['qlearning', 'dynaq', 'dynaq_tuned', 'a2c_td']
ALGO_COLORS = {k: v[1] for k,v in ALGOS.items()}
ALGO_LABELS = {k: v[0] for k,v in ALGOS.items()}
ALGO_STYLES = {'qlearning':'-','dynaq':'--','dynaq_tuned':'-.','a2c_td':':'}

# ── A) Per-env × per-decay: all 4 algos on same plot ─────────────────────────
print("[A] Per-env per-decay comparison...")
for env_id, env_label in ENVS:
    for decay in DECAYS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Algorithm Comparison — {env_label}  [{DECAY_LABELS[decay]}]',
                     fontsize=13, fontweight='bold')
        ax_r, ax_sr, ax_eps = axes
        for algo in ALGO_LIST:
            st = load(algo, env_id, decay)
            if st is None: continue
            c   = ALGO_COLORS[algo]
            ls  = ALGO_STYLES[algo]
            lbl = ALGO_LABELS[algo]
            ax_r.plot(smooth(st['rewards']),  color=c, ls=ls, lw=1.8, label=lbl)
            ax_sr.plot(smooth(st['success']), color=c, ls=ls, lw=1.8, label=lbl)
            ax_eps.plot(st['epsilons'],        color=c, ls=ls, lw=1.2, alpha=0.85, label=lbl)
        ax_r.axhline(0, color='k', ls='--', lw=0.6, alpha=0.4)
        for ax, ylabel, title in [(ax_r,'Total Reward (avg 100)','Training Curve'),
                                   (ax_sr,'Success Rate (avg 100)','Success Rate'),
                                   (ax_eps,'Epsilon ε','Exploration Rate')]:
            ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax_sr.set_ylim(-0.05, 1.05); ax_eps.set_ylim(0, 1.05)
        plt.tight_layout()
        savefig(fig, f'{OUT}/compare__{env_id}__{decay}.png')

# ── B) Summary heatmap per algo (algo as rows, env as cols) ──────────────────
print("[B] Multi-algo heatmaps...")
for decay in DECAYS:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'All Algorithms — {DECAY_LABELS[decay]}  |  Success Rate & Avg Reward',
                 fontsize=13, fontweight='bold')
    sr_mat = np.full((len(ALGO_LIST), len(ENVS)), np.nan)
    r_mat  = np.full((len(ALGO_LIST), len(ENVS)), np.nan)
    for i, algo in enumerate(ALGO_LIST):
        for j, (env_id, _) in enumerate(ENVS):
            sm = summary(load(algo, env_id, decay))
            if sm: sr_mat[i,j]=sm['sr']; r_mat[i,j]=sm['avg_r']

    algo_labels_list = [ALGO_LABELS[a] for a in ALGO_LIST]
    env_labels_list  = [e[1] for e in ENVS]

    for ax, mat, title, fmt in [(axes[0],sr_mat,'Success Rate','sr'),(axes[1],r_mat,'Avg Reward','r')]:
        vmin=np.nanmin(mat); vmax=np.nanmax(mat)
        im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ENVS))); ax.set_xticklabels(env_labels_list, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(ALGO_LIST))); ax.set_yticklabels(algo_labels_list, fontsize=9)
        ax.axvline(2.5, color='white', lw=2.5); ax.set_title(title, fontsize=11, fontweight='bold')
        for i in range(len(ALGO_LIST)):
            for j in range(len(ENVS)):
                v=mat[i,j]
                if np.isnan(v): continue
                txt=f'{v:.0%}' if fmt=='sr' else f'{v:.2f}'
                nv=(v-vmin)/(vmax-vmin+1e-9)
                ax.text(j,i,txt,ha='center',va='center',fontsize=8,fontweight='bold',
                        color='white' if nv<0.35 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    savefig(fig, f'{OUT}/heatmap__{decay}.png')

# ── C) Best decay per algo per env (winner chart) ────────────────────────────
print("[C] Best decay winner chart...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Best Performing Decay Strategy — All Algorithms & Environments',
             fontsize=13, fontweight='bold')
for idx, (env_id, env_label) in enumerate(ENVS):
    ax = axes[idx//3, idx%3]
    ax.set_title(env_label, fontsize=10, fontweight='bold')
    x = np.arange(len(ALGO_LIST))
    width = 0.2
    for di, decay in enumerate(DECAYS):
        srs = []
        for algo in ALGO_LIST:
            sm = summary(load(algo, env_id, decay))
            srs.append(sm['sr'] if sm else 0)
        bars = ax.bar(x + di*width, srs, width, label=DECAY_LABELS[decay],
                      color=DECAY_COLORS[decay], alpha=0.85)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([ALGO_LABELS[a].replace(' ','  ') for a in ALGO_LIST], fontsize=7, rotation=15)
    ax.set_ylabel('Success Rate (last 500 eps)')
    ax.set_ylim(0, 1.1); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, f'{OUT}/winner_chart.png')

# ── D) Scaling: success rate vs env size (all algos, best decay) ─────────────
print("[D] Scaling analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Scalability: Success Rate vs Environment Size',
             fontsize=13, fontweight='bold')
sizes = ['5×5', '8×8', '16×16']
for col, (group, env_group) in enumerate([('Empty Environments', ENVS_EMPTY),
                                           ('DoorKey Environments', ENVS_DOORKEY)]):
    ax = axes[col]
    for algo in ALGO_LIST:
        # use linear decay as common baseline for fair comparison
        srs = []
        for env_id, _ in env_group:
            sm = summary(load(algo, env_id, 'linear'))
            srs.append(sm['sr'] if sm else 0)
        ax.plot(sizes, srs, color=ALGO_COLORS[algo], ls=ALGO_STYLES[algo],
                lw=2.2, marker='o', ms=9, label=ALGO_LABELS[algo])
    ax.set_title(group); ax.set_xlabel('Environment Size')
    ax.set_ylabel('Success Rate — Linear Decay (last 500 eps)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)
plt.tight_layout()
savefig(fig, f'{OUT}/scaling_analysis.png')

# ── E) First success episode comparison ──────────────────────────────────────
print("[E] First success comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sample Efficiency: First Success Episode (Linear Decay)',
             fontsize=13, fontweight='bold')
for col, (group, env_group) in enumerate([('Empty', ENVS_EMPTY), ('DoorKey', ENVS_DOORKEY)]):
    ax = axes[col]
    env_labels_g = [e[1] for e in env_group]
    x = np.arange(len(env_group))
    width = 0.2
    for ai, algo in enumerate(ALGO_LIST):
        firsts = []
        for env_id, _ in env_group:
            sm = summary(load(algo, env_id, 'linear'))
            firsts.append(sm['first'] if sm and sm['first'] else 5000)
        ax.bar(x + ai*width, firsts, width, label=ALGO_LABELS[algo],
               color=ALGO_COLORS[algo], alpha=0.85)
    ax.set_xticks(x + width*1.5); ax.set_xticklabels(env_labels_g, fontsize=9)
    ax.set_title(f'{group} Environments'); ax.set_ylabel('First Success Episode (lower=better)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, f'{OUT}/first_success.png')

# ── Print full comparison table ───────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"{'Algo':<18} {'Env':<16} {'Decay':<14} {'SR':>6} {'AvgR':>7} {'First':>8} {'PeakSR':>8}")
print(f"{'-'*90}")
for algo in ALGO_LIST:
    for env_id, env_label in ENVS:
        for decay in DECAYS:
            sm = summary(load(algo, env_id, decay))
            if not sm: continue
            fe = f"ep{sm['first']}" if sm['first'] else 'Never'
            print(f"  {ALGO_LABELS[algo]:<16} {env_label:<16} {DECAY_LABELS[decay]:<14} "
                  f"{sm['sr']:>5.0%} {sm['avg_r']:>7.3f} {fe:>8} {sm['peak_sr']:>7.0%}")
    print()

print(f"\nDone! All comparison plots saved to {OUT}/")
