from paraview.simple import ADIOS2VTXReader, servermanager, Slice, ResampleToImage, Transform
from vtk.util.numpy_support import vtk_to_numpy

import numpy as np
import matplotlib.pyplot as plt


# --- Paper‐style settings ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


# --- Reference wave ---
frequency = 0.5e6  # Hz
amplitude = 60000  # peak
c         = 1500   # m/s

# --- Data loading & reference ---
field     = "u"
file1 = '../examples/spherical_transducer_gpu/analytical3d.bp'
axis         = "X"              # "X","Y" or "Z"
value        = 0.0              # slice location
dims         = [1, 500, 500]    # [nx,ny,1]
    
def reader(bp):
    r = ADIOS2VTXReader(FileName=[bp])
    r.UpdatePipeline()
    return r

def slicer(r):
    s = Slice(Input=r)
    s.UpdatePipeline()
    bounds = s.GetDataInformation().GetBounds()
    return s, bounds

def flip_z(input):
    t = Transform(Input=input)
    t.Transform.Scale = [1, 1, -1]
    t.UpdatePipeline()
    return t


# file1 = '../examples/spherical_transducer_gpu/planar.bp'
# file1 = '../examples/hex_gpu/planar_hex_2.bp'
# file1 = '../examples/hex_gpu/planar_hex_4.bp'
# file1 = '../examples/hex_gpu/output.bp'
# file1 = '../examples/spherical_transducer_gpu/output.bp'
file1 = '../examples/spherical_transducer_gpu/analytical3d.bp'
# file1 = '../examples/spherical_transducer_gpu/small_spherical.bp'
# file1 = '../build/output_final.bp'

r1 = reader(file1)

t  = r1.TimestepValues if isinstance(r1.TimestepValues, float) else r1.TimestepValues[-1]
print("Timestep:", t)

r1.UpdatePipeline(t)
b1 = r1.GetDataInformation().GetBounds()
s1, b1 = slicer(r1)
b1 = [0.0 if abs(b) < 1e-15 else b for b in b1]
print("Bounds 1:", b1)

def resample(s):
    r = ResampleToImage(Input=s)
    r.SamplingBounds     = b1
    r.SamplingDimensions = dims
    r.UpdatePipeline()
    return r

res = resample(s1)
d = servermanager.Fetch(res)
u1 = vtk_to_numpy(d.GetPointData().GetArray(field)).reshape(dims[1], dims[2])




omega     = 2*np.pi*frequency
k         = omega/c
xmin, xmax= b1[2], b1[3]
x         = np.linspace(xmin, xmax, dims[1])
u_ref     = amplitude * np.cos(k * x - omega * (t + 0.17e-6))

u1_kPa    = u1[dims[1]//2, :] * 1e-3
u_ref_kPa = u_ref * 1e-3

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, u_ref_kPa, label='Analytical', color='black')
ax.scatter(x, u1_kPa,    label='Computed', marker="x", s=10)


ax.set_xlabel('x (m)')
ax.set_ylabel('Pressure (kPa)')
ax.legend(frameon=True, loc='upper right')
ax.grid(False)

plt.tight_layout()
plt.savefig('wave_comparison3d.png')