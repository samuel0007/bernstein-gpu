import gmsh, math

gmsh.initialize()
gmsh.model.add("spherical_transducer")

# Parameters
# Rcurv = 0.064         # radius of curvature
# A     = 0.064         # aperture diameter
# Lbox  = 0.120         # backing‐box length
Rcurv = 0.016         # radius of curvature
A     = 0.016         # aperture diameter
Lbox  = 0.03         # backing‐box length

frequency = 0.5e6 # [Hz]
speedOfSound = 1500 # [m/s]
wavelength = speedOfSound / frequency # [m]
elementsPerWavelength = 2.4 # [elements]
inflow_marker = 1
outflow_marker = 2
volume_marker = 1

h = wavelength / elementsPerWavelength

a = A/2.0
s = Rcurv - math.sqrt(Rcurv**2 - a**2)

# 1) full sphere centered so its cap is at z=0
sphere_tag = gmsh.model.occ.addSphere(0, 0, -Rcurv, Rcurv)

# 2) cap‐box: x,y∈[−a,a], z∈[−s,0]
cap_box_tag = gmsh.model.occ.addBox(-a, -a, -s, 2*a, 2*a, s)

gmsh.model.occ.synchronize()

# 3) spherical cap = intersection(sphere, cap_box)
cap_list, _ = gmsh.model.occ.intersect(
    [(3, sphere_tag)], [(3, cap_box_tag)]
)
cap = cap_list[0]   # (3, tag)
# 4) backing box: x,y∈[−a,a], z∈[−Lbox,−s]
back_cyl = gmsh.model.occ.addCylinder(
        0, 0, -Lbox,       # base center
        0, 0, Lbox - s,    # axis vector
        a                        # radius
    )
gmsh.model.occ.synchronize()

# # 5) fuse cap + backing
fuse_list, _ = gmsh.model.occ.fuse([cap], [(3, back_cyl)])
fused = fuse_list[0]   # (3, tag)

gmsh.model.occ.synchronize()

# 6) extract all surface faces of the fused volume
faces = gmsh.model.getBoundary([(3, fused[1])], oriented=False, recursive=False)
all_face_tags = [f[1] for f in faces]

# 7) identify inlet: the face whose z_max ≈ 0
inlet_tag = None
tol = 1e-6
for tag in all_face_tags:
    # returns [xmin, ymin, zmin, xmax, ymax, zmax]
    bb = gmsh.model.getBoundingBox(2, tag)
    zmin, zmax = bb[2], bb[5]
    if abs(zmax) < tol and zmin < tol:
        inlet_tag = tag
        break
if inlet_tag is None:
    raise RuntimeError("Could not find inlet face at z≈0")

# 8) classify outflow = all other faces
outflow_tags = [tag for tag in all_face_tags if tag != inlet_tag]

# 9) physical groups & mesh
gmsh.model.addPhysicalGroup(2, [inlet_tag], inflow_marker)
gmsh.model.addPhysicalGroup(2, outflow_tags, outflow_marker)
gmsh.model.addPhysicalGroup(3, [fused[1]], volume_marker)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
gmsh.model.mesh.generate(3)

# gmsh.fltk.run()

gmsh.write("data/transducer.msh")

gmsh.finalize()
