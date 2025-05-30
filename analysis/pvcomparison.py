from paraview.simple import ADIOS2VTXReader, servermanager, Slice, ResampleToImage, Transform
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt

field        = "u"              
axis         = "X"              # "X","Y" or "Z"
value        = 0.0              # slice location
dims         = [1, 300, 300]    # [nx,ny,1]
    
def reader(bp):
    r = ADIOS2VTXReader(FileName=[bp])
    r.UpdatePipeline()
    return r

def slicer(r):
    s = Slice(Input=r)
    s.UpdatePipeline()
    bounds = s.GetDataInformation().GetBounds()
    return s, bounds



# file1 = '../examples/spherical_transducer_gpu/spherical_p2_c5.bp'
# file2 = '../examples/hex_gpu/spherical_hex.bp'

file2 = '../examples/spherical_transducer_gpu/planar.bp'
file1 = '../examples/hex_gpu/planar_hex.bp'
# file2 = '../examples/hex_gpu/planar_hex_4.bp' # this is not the same domain

# file1 = '../examples/hex_gpu/spherical_small_p4.bp'
# file2 = '../examples/spherical_transducer_gpu/small_spherical.bp'

# file1 = '../examples/hex_gpu/spherical_smaller_p4_24_output.bp'
# file2 = '../examples/spherical_transducer_gpu/small_spherical_output.bp'

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
})

# file2 = '../build/output_final.bp'

r1, r2 = reader(file1), reader(file2)

times1 = r1.TimestepValues[-5:]
times2 = r2.TimestepValues

# for each t1, pick t2 with minimum |t2−t1|
closest = [min(times2, key=lambda t2: abs(t2 - t1)) for t1 in times1]
times2 = closest
for t1, t2 in zip(times1, times2):
    print(f"t1={t1:.6f} → closest t2={t2:.6f}")

u1_stack = np.empty((len(times1), np.prod(dims)))
u2_stack = np.empty_like(u1_stack)

b1 = r1.GetDataInformation().GetBounds()
b2 = r2.GetDataInformation().GetBounds()

s1, b1 = slicer(r1)
b1 = [0.0 if abs(b) < 1e-15 else b for b in b1]
print("Bounds 1:", b1)
s2, b2 = slicer(r2)
b2 = [0.0 if abs(b) < 1e-15 else b for b in b2]
print("Bounds 2:", b2)

if axis=="X":
    sb = [value,value, min(b1[2],b2[2]), max(b1[3],b2[3]), min(b1[4],b2[4]), max(b1[5],b2[5])]
elif axis=="Y":
    sb = [min(b1[0],b2[0]), max(b1[1],b2[1]), value,value, min(b1[4],b2[4]), max(b1[5],b2[5])]
else:
    sb = [min(b1[0],b2[0]), max(b1[1],b2[1]), min(b1[2],b2[2]), max(b1[3],b2[3]), value,value]

print("Bounds:", sb)

def resample(s):
    r = ResampleToImage(Input=s)
    r.SamplingBounds     = sb
    r.SamplingDimensions = dims
    return r


for i, (t1, t2) in enumerate(zip(times1, times2)):
    print("Processing timestep", i+1, "of", len(times1))
    # load that timestep
    r1.UpdatePipeline(t1)
    r2.UpdatePipeline(t2)
    # slice & resample
    s1, _ = slicer(r1);   s2, _ = slicer(r2)
    res1 = resample(s1);  res2 = resample(s2)
    res1.UpdatePipeline(); res2.UpdatePipeline()
    # fetch arrays
    d1 = servermanager.Fetch(res1)
    d2 = servermanager.Fetch(res2)
    u1 = vtk_to_numpy(d1.GetPointData().GetArray(field))
    # u1 = np.flip(u1.reshape(dims[2],dims[1]), axis=0)
    # u1_stack[i] = u1.flatten()
    u1_stack[i] = u1
    
    # plot
    plt.imshow(u1_stack[i].reshape(dims[2],dims[1]), cmap='jet', aspect='auto')
    plt.savefig(f'u1_timestep_{i+1}.png')
    plt.close()

    u2_stack[i] = vtk_to_numpy(d2.GetPointData().GetArray(field))
    plt.imshow(u2_stack[i].reshape(dims[2], dims[1]), cmap='jet', aspect='auto')
    plt.savefig(f'u2_timestep_{i+1}.png')
    plt.close()
    

nt, npts = u1_stack.shape

# compute sampling interval (assumes uniform sampling)
dt = np.mean(np.diff(times1))

# FFT along time axis
fft1 = np.fft.rfft(u1_stack, axis=0)
fft2 = np.fft.rfft(u2_stack, axis=0)
freqs = np.fft.rfftfreq(nt, dt)

f0 = 0.5e6  # Hz
idx = np.argmin(np.abs(freqs - f0))

# amplitude = 2*|FFT at f0|/nt
amp1_fft = 2 * np.abs(fft1[idx, :]) / nt
amp2_fft = 2 * np.abs(fft2[idx, :]) / nt

# reshape to spatial grid

amp1 = amp1_fft.reshape(dims[2], dims[1])
amp2 = amp2_fft.reshape(dims[2], dims[1])
# --- Compute relative errors ---
rel_L2 = np.linalg.norm(amp2-amp1) / np.linalg.norm(amp1)
rel_L1 = np.sum(np.abs(amp2-amp1)) / np.sum(np.abs(amp1))
    
# plt.imshow(amp1, cmap='jet')
# plt.colorbar(label='Amplitude (u1)')
# plt.title('Amplitude from File 1')
# plt.savefig('amp1.png')
# plt.close()

# plt.imshow(amp2, cmap='jet')
# plt.colorbar(label='Amplitude (u2)')
# plt.title('Amplitude from File 2')
# plt.savefig('amp2.png')
# plt.close()
    
M, N = amp1.shape
mid = N // 2

# build composite array
comp = np.empty_like(amp1)
comp[:, :mid] = amp1[:, :mid]
comp[:, mid:] = amp2[:, mid:]

# flip
comp = np.flip(comp, axis=0)

# # plot
# plt.figure(figsize=(6, 5))
# im = plt.imshow(comp, origin='lower', aspect='auto', cmap="turbo")
# # plt.title('Left half: File 1 | Right half: File 2')
# plt.colorbar(im, label='Amplitude')
# plt.savefig("composite.png")
# plt.close()



# --- Professional plot ---
fig, ax = plt.subplots(figsize=(6, 5))

ymin, ymax = sb[2], sb[3]
zmin, zmax = sb[4], sb[5]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    comp,
    origin='lower',
    aspect='auto',
    cmap='turbo',
    extent=[ymin, ymax, zmin, zmax]
)

y_mid = 0.5 * (ymin + ymax)
ax.axvline(x=y_mid, color='white', linestyle='--', linewidth=1.0)

# annotate errors 
txt = (f'Rel. $L_2$ error: {100 * rel_L2:.2f} %\n'
       f'Rel. $L_1$ error: {100 * rel_L1:.2f} %')
ax.text(0.02, 0.98, txt, transform=ax.transAxes,
        va='top', ha='left', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

ax.set_xlabel('y (m)')
ax.set_ylabel('z (m)')
# ax.set_title('FFT Amplitude Composite\n(Left: File 1, Right: File 2)', pad=6)

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Amplitude')

# remove top/right spines
# for spine in ('top', 'right'):
#     ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('composite.png')
plt.close()