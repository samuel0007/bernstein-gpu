import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(filename):
    measured = np.nan
    baseline = np.nan
    with open(filename, 'r') as f:
        for line in f:
            m1 = re.search(r'Mat-free action Gdofs/s:\s*([0-9.]+)', line)
            if m1:
                measured = float(m1.group(1))
            m2 = re.search(r'Baseline Mat-free action Gdofs/s:\s*([0-9.]+)', line)
            if m2:
                baseline = float(m2.group(1))
    return measured, baseline

# set P to your actual polynomial orders, e.g. 1–5
P = list(range(1, 12))
meas, base = zip(*(parse_log_file(f'log_{i}.txt') for i in P))

plt.plot(P, meas, marker='o', label='Sum Facto')
plt.plot(P, base, linestyle='--', marker='x', label='Baseline')
plt.xlabel('Polynomial order P')
plt.ylabel('GDoFs/s')
plt.title('Mat-free performance on A30')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('mat_free_A30.png')
