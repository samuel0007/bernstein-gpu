import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(filename):
    sf_otf = np.nan
    sf = np.nan
    baseline = np.nan
    with open(filename, 'r') as f:
        for line in f:
            # m1 = re.search(r'SF OTF Mat-free action Gdofs/s:\s*([0-9.]+)', line)
            # if m1:
            #     sf_otf = float(m1.group(1))
            m2 = re.search(r'Baseline Mat-free action Gdofs/s:\s*([0-9.]+)', line)
            if m2:
                baseline = float(m2.group(1))
            m3 = re.search(r'SF Mat-free action Gdofs/s:\s*([0-9.]+)', line)
            if m3:
                sf = float(m3.group(1))
    return sf, sf_otf, baseline

P = list(range(2, 10))
sf, sf_otf, base = zip(*(parse_log_file(f'log_float32_{i}.txt') for i in P))

plt.plot(P, sf, marker='o', label='Sum Facto')
# plt.plot(P, sf_otf, marker='>', label='Sum Facto On the fly')
plt.plot(P, base, linestyle='--', marker='x', label='Baseline')
plt.xlabel('Polynomial order P')
plt.ylabel('GDoFs/s')
plt.title('Mat-free performance on RTX A 6000 \n 3D stiffness on tet mesh, shared mem optimisation, 32bit')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('rtx_mat_free_stiff_32B.png')
