import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(filename):
    pcg = np.nan; cg = np.nan
    with open(filename, 'r') as f:
        for line in f:
            m = re.search(r'Number of GPU PCG iterations\s*([0-9.]+)', line)
            if m: pcg = float(m.group(1))
            m = re.search(r'Number of GPU CG iterations\s*([0-9.]+)', line)
            if m: cg  = float(m.group(1))
    return pcg, cg

P         = np.arange(2, 7)
b_pcg, b_cg = zip(*(parse_log_file(f'log_bernstein_{i}.txt') for i in P))
g_pcg, g_cg = zip(*(parse_log_file(f'log_gll_{i}.txt')       for i in P))

# --- Use serif (e.g. Times) and Computer Modern for math ---
plt.rcParams.update({
    'font.family':      'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# grayscale levels
colors = {'bernstein': '0.', 'gll': '0.5'}

# plot with distinct markers and line styles
ax.plot(P, b_pcg, linestyle='-',  marker='D', markersize=6, linewidth=1,
        color=colors['bernstein'], label='Bernstein PCG')
ax.plot(P, b_cg,  linestyle='--', marker='s', markersize=6, linewidth=1,
        color=colors['bernstein'], label='Bernstein CG')
ax.plot(P, g_pcg, linestyle='-',  marker='o', markersize=6, linewidth=1,
        color=colors['gll'],       label='warpedGLL PCG')
ax.plot(P, g_cg,  linestyle='--', marker='^', markersize=6, linewidth=1,
        color=colors['gll'],       label='warpedGLL CG')

# axes labels and ticks
ax.set_xlabel('Polynomial order $P$', fontsize=14)
ax.set_ylabel('Number of CG iterations', fontsize=14)
ax.set_xticks(P)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.minorticks_on()

# grid: major and minor
ax.grid(which='major', linestyle='-',  linewidth=0.5)
ax.grid(which='minor', linestyle=':',  linewidth=0.3)

# legend outside plot
ax.legend(frameon=True, fontsize=12, loc='best',)

# plt.xlim(1.5, 6)
plt.ylim(bottom=0)


fig.tight_layout()
fig.savefig('3d_cg.png', dpi=300)
