"""
run_all.py — Chạy tất cả notebooks theo thứ tự để sinh toàn bộ plots
Usage: python notebooks/run_all.py
"""
import subprocess, sys, os

scripts = [
    '01_visualize_qlearning.py',
    '02_visualize_dynaq.py',
    '03_visualize_dynaq_tuned.py',
    '04_visualize_a2c.py',
    '05_compare_all.py',
]

base = os.path.dirname(os.path.abspath(__file__))
total = 0
for s in scripts:
    path = os.path.join(base, s)
    print(f"\n{'='*60}")
    print(f"Running: {s}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, path], cwd=base)
    if result.returncode != 0:
        print(f"ERROR in {s}")
        sys.exit(1)
    total += 1

print(f"\n{'='*60}")
print(f"Done! {total}/{len(scripts)} scripts completed.")
print(f"Plots saved to: {os.path.join(os.path.dirname(base), 'plots')}/")
