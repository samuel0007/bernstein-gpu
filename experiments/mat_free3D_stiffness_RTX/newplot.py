import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

PALETTE_HEX = ["#E88B7A", "#F6F0DB", "#9FC7C3", "#8A6C7D", "#253E48"]
num_pat = r'([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)'

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120
})


def parse_val(fname, label):
    """Return throughput for given label regex."""
    if not Path(fname).exists():
        return np.nan
    with open(fname, 'r') as f:
        for line in f:
            m = re.search(rf'^{label} Mat-free action Gdofs/s:\s*{num_pat}', line)
            if m:
                return float(m.group(1))
    return np.nan

# ---- inputs ----
P = list(range(2, 10))
prefix_tuned = "log_float32_"
prefix_aln   = "log_aligned_float32_"
TOTAL_GDOFS = None  # e.g. 1.0e9 for true seconds; None -> relative

tuned_vals = []
aligned_vals = []
for p in P:
    tuned_vals.append(parse_val(f"{prefix_tuned}{p}.txt", "Tuned"))
    aligned_vals.append(parse_val(f"{prefix_aln}{p}.txt", "Tuned\\+Aligned"))

tuned_vals   = np.array(tuned_vals, dtype=float)
aligned_vals = np.array(aligned_vals, dtype=float)

# Speedup aligned vs tuned
speedup = aligned_vals / tuned_vals

# Convert to time if TOTAL_GDOFS provided
def to_time(thr):
    if TOTAL_GDOFS is None:
        return 1.0 / thr
    return TOTAL_GDOFS / thr

time_tuned   = to_time(tuned_vals)
time_aligned = to_time(aligned_vals)

# ---- plot ----
fig, ax1 = plt.subplots(figsize=(6, 4))

ax1.plot(P, tuned_vals, marker='o', color=PALETTE_HEX[0], label='Tuned')
ax1.plot(P, aligned_vals, marker='s', color=PALETTE_HEX[2], label='Tuned+Aligned')

ax1.set_xlabel('Polynomial order P')
ax1.set_ylabel('GDoFs/s')
ax1.grid(True, which='both', linestyle=':', linewidth=0.6)

# Right axis: time
# ax2 = ax1.twinx()
# ax2.plot(P, time_tuned, alpha=0)
# ax2.set_ylabel('Time [s]' if TOTAL_GDOFS is not None else 'Relative time (1 / GDoFs/s)')

# Annotate speedups
for p, y, s in zip(P, aligned_vals, speedup):
    if np.isfinite(y) and np.isfinite(s):
        ax1.text(p, y + 0.07, f"×{s:.2f}",
                 ha='center', va='bottom', fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='black',
                           boxstyle='round,pad=0.2', linewidth=0.6))

plt.ylim(top=1.09 * max(np.nanmax(tuned_vals), np.nanmax(aligned_vals)))
ax1.legend(loc='best')
# plt.title('Mat-free performance (float32): Tuned vs Tuned+Aligned')
plt.tight_layout()
plt.savefig('tuned_vs_tuned_aligned_stiffness_float32.png', dpi=300)
plt.show()
