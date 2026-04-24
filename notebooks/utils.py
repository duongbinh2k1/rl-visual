"""
utils.py — Shared utilities for all notebooks
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(ROOT, 'data')
PLOTS_DIR  = os.path.join(ROOT, 'plots')

# ── Config ────────────────────────────────────────────────────────────────────
ENVS = [
    ('empty_5x5',     'Empty 5×5'),
    ('empty_8x8',     'Empty 8×8'),
    ('empty_16x16',   'Empty 16×16'),
    ('doorkey_5x5',   'DoorKey 5×5'),
    ('doorkey_8x8',   'DoorKey 8×8'),
    ('doorkey_16x16', 'DoorKey 16×16'),
]
ENVS_EMPTY   = ENVS[:3]
ENVS_DOORKEY = ENVS[3:]

DECAYS = ['linear', 'exp', 'rbed_linear', 'rbed_exp']
DECAY_LABELS = {
    'linear':      'Linear Decay',
    'exp':         'Exponential Decay',
    'rbed_linear': 'RBED + Linear',
    'rbed_exp':    'RBED + Exponential',
}
DECAY_COLORS = {
    'linear':      '#2266CC',
    'exp':         '#22AA55',
    'rbed_linear': '#CC4422',
    'rbed_exp':    '#AA22AA',
}
DECAY_STYLES = {
    'linear':      '-',
    'exp':         '--',
    'rbed_linear': ':',
    'rbed_exp':    '-.',
}

ALGOS = {
    'qlearning':   ('Q-Learning',          'royalblue'),
    'dynaq':       ('Dyna-Q (n=5)',         'green'),
    'dynaq_tuned': ('Dyna-Q Tuned (n=20)', 'teal'),
    'a2c_td':      ('A2C (TD)',             'tomato'),
}

METRICS = ['rewards', 'steps', 'epsilons', 'success', 'avg_q']

# ── Load ──────────────────────────────────────────────────────────────────────
def load(algo, env, decay):
    folder = os.path.join(DATA_DIR, algo)
    paths  = {m: os.path.join(folder, f'{algo}__{env}__{decay}__{m}.npy')
              for m in METRICS}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return {m: np.load(paths[m]) for m in METRICS}

def smooth(x, w=100):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

def summary(st, last_n=500):
    if st is None: return None
    r, s, steps = st['rewards'], st['success'], st['steps']
    first = next((i for i,v in enumerate(s) if v > 0), None)
    peak  = float(np.max(np.convolve(s, np.ones(100)/100, 'valid'))) if len(s)>=100 else float(np.mean(s))
    return {
        'sr':        float(np.mean(s[-last_n:])),
        'sr100':     float(np.mean(s[-100:])),
        'avg_r':     float(np.mean(r[-last_n:])),
        'best_r':    float(np.max(r)),
        'avg_steps': float(np.mean(steps[-last_n:])),
        'first':     first,
        'peak_sr':   peak,
        'n':         len(r),
    }

def savefig(fig, path, dpi=130):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
