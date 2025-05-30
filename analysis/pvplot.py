from paraview.simple import ADIOS2VTXReader, servermanager, Slice, ResampleToImage, Transform
from vtk.util.numpy_support import vtk_to_numpy

import numpy as np
import matplotlib.pyplot as plt

field        = "u"              
axis         = "X"              # "X","Y" or "Z"
value        = 0.0              # slice location
dims         = [1, 141, 241]    # [nx,ny,1]
    
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
# file1 = '../examples/hex_gpu/spherical_small_p4.bp'
file1 = '../examples/hex_gpu/spherical_smaller_p4_24.bp'


# file1 = '../examples/spherical_transducer_gpu/output.bp'
# file1 = '../examples/spherical_transducer_gpu/analytical3d.bp'
# file1 = '../examples/spherical_transducer_gpu/small_spherical.bp'
# file1 = '../build/output_final.bp'

r1 = reader(file1)


if type(r1.TimestepValues) == float:
    t = r1.TimestepValues
else:
    t = r1.TimestepValues[-1]
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
    return r

res = resample(s1)
res.UpdatePipeline()
d = servermanager.Fetch(res)
u1 = vtk_to_numpy(d.GetPointData().GetArray(field))

# all 0 are nans
# u1[np.abs(u1) < 1] = np.nan

u1 = u1.reshape(dims[2], dims[1])

# mask = np.isnan(u1)
# indices = distance_transform_edt(mask, return_distances=False, return_indices=True)

# fill NaNs: for each (i,j) that was NaN, take u1 at the nearest valid index
# u1_filled = u1[tuple(indices)]

# interpolate all nans to nearest neighbor

plt.imshow(u1, cmap='jet', aspect='auto')
plt.colorbar(label='Wave (u1)')
plt.savefig('u1.png')