import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn

# --- 1. Physical parameters ---
c = 1480.0       # [m/s]
rho = 1000.0     # [kg/m^3]
beta = 3.5       # nonlinearity coeff
f = 0.1e6        # Hz
P0 = 80e6         # Pa (5 MPa)

alpha_fund = 0.22  # [Np/m] absorption at fundamental

omega = 2*np.pi*f
k = omega/c

# shock formation distance
x_s = rho * c**3 / (beta * omega * P0)
shock_dist = x_s   # same definition

# validation point: 1 cm
x_validate = 0.01

# dimensionless z
z_validate = x_validate / shock_dist

# Goldberg number
Gamma = 1 / (alpha_fund * shock_dist)

print("--- Validation Benchmark Setup ---")
print(f"P0 = {P0/1e6:.2f} MPa")
print(f"Shock distance = {shock_dist*100:.2f} cm")
print(f"x_validate = {x_validate*100:.2f} cm")
print(f"z = {z_validate:.2f}")
print(f"Gamma = {Gamma:.2f}")
print("----------------------------------")

# --- 2. Waveform at x_validate ---
T = 1.0/f
t_arrival = x_validate / c
t = t_arrival + np.linspace(0.0, T, 5000, endpoint=False)
Nmax = 100

p_analytical = np.zeros_like(t)
harmonics_analytical_p = np.zeros(Nmax)

alpha1 = alpha_fund  # attenuation coeff for n=1

    
p_analytical = np.zeros_like(t)
harmonics_analytical_p = np.zeros(Nmax)

for n in range(1, Nmax+1):
    coeff = (2*P0/(n*z_validate)) * jn(n, n*z_validate) * np.exp(-(n**2)*alpha1*x_validate)
    harmonics_analytical_p[n-1] = abs(coeff)
    # enforce sine series for 0→0 boundary
    p_analytical += coeff * np.sin(n*omega*(t - t_arrival))


# --- 3. Plot waveform ---
plt.figure(figsize=(8,5))
plt.plot((t - t_arrival)*1e6, p_analytical/1e6, 'k-', lw=2)
plt.xlabel("Retarded Time [µs]")
plt.ylabel("Pressure [MPa]")
plt.title(f"Analytical Waveform at x={x_validate*100:.1f} cm (z={z_validate:.2f})")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig("analytical_benchmark.png", dpi=300)
plt.show()

# --- 4. Harmonics ---
plt.figure(figsize=(8,5))
n_plot = min(20, Nmax)
plt.stem(np.arange(1,n_plot+1)*f/1e6, harmonics_analytical_p[:n_plot]/1e6, basefmt=" ")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Amplitude [MPa]")
plt.title(f"Harmonic spectrum at x={x_validate*100:.1f} cm")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig("analytical_harmonics.png", dpi=300)
plt.show()
