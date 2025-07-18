import gmsh, math

gmsh.initialize()
gmsh.model.add("spherical_transducer")



# BP1-small / BP2-small
# frequency = 0.1e6 # [Hz]
# materials_sound_speed = [1500] # [m/s]
# Lcap = 0.12         # Length of the cap box [m]
# layers_height = []
# layers_volume_markers = []
# elementsPerWavelength = 2.4 # [elements]
# Rcurv = 0.064       # radius of curvature
# A     = 0.064       # aperture diameter

# BP1 / BP2
# frequency = 0.5e6 # [Hz]
# materials_sound_speed = [1500] # [m/s]
# Lcap = 0.12         # Length of the cap box [m]
# layers_height = []
# layers_volume_markers = []
# elementsPerWavelength = 2.4 # [elements]
# Rcurv = 0.064       # radius of curvature
# A     = 0.064       # aperture diameter

# BP3
# frequency = 0.5e6 # [Hz]
# materials_sound_speed = [1500, 2800] # [m/s]
# Lcap = 0.03         # Length of the cap box [m]
# layers_height = [0.0065, 0.0835]
# layers_volume_markers = [2, 1]
# elementsPerWavelength = 2.4 # [elements]
# Rcurv = 0.064       # radius of curvature
# A     = 0.064       # aperture diameter

# BP3-small
frequency = 0.1e6 # [Hz]
materials_sound_speed = [1500, 2800] # [m/s]
Lcap = 0.03         # Length of the cap box [m]
layers_height = [0.0065, 0.0835]
layers_volume_markers = [2, 1]
elementsPerWavelength = 2.4 # [elements]
Rcurv = 0.064       # radius of curvature
A     = 0.064       # aperture diameter

# BP4
# frequency = 0.5e6 # [Hz]
# # Water: 1, Skin: 2, Cortical: 3, Trabecular: 4, Brain: 5
# materials_sound_speed = [1500, 1610, 2800, 2300, 1560] # [m/s]
# Lcap = 0.026        # Length of the cap box [m]
# layers_height = [0.004, 0.0015, 0.004, 0.001, 0.0835]
# layers_volume_markers = [2, 3, 4, 3, 5]
# elementsPerWavelength = 2.4 # [elements]
# Rcurv = 0.064       # radius of curvature
# A     = 0.064       # aperture diameter

# BP4-small
# frequency = 0.1e6 # [Hz]
# # Water: 1, Skin: 2, Cortical: 3, Trabecular: 4, Brain: 5
# materials_sound_speed = [1500, 1610, 2800, 2300, 1560] # [m/s]
# Lcap = 0.026        # Length of the cap box [m]
# layers_height = [0.004, 0.0015, 0.004, 0.001, 0.0835]
# layers_volume_markers = [2, 3, 4, 3, 5]
# elementsPerWavelength = 2.4 # [elements]
# Rcurv = 0.064       # radius of curvature
# A     = 0.064       # aperture diameter

cap_volume_marker = 1
inflow_marker, outflow_marker = 1, 2


materials_wavelength = [sound_speed / frequency for sound_speed in materials_sound_speed]
materials_h = [wavelength / elementsPerWavelength for wavelength in materials_wavelength]
Nmaterials = len(materials_sound_speed)

a = A/2.0 # Aperture radius
s = Rcurv - math.sqrt(Rcurv**2 - a**2) # Height of cap

# 1) full sphere centered so its cap is at z=0
sphere_tag = gmsh.model.occ.addSphere(0, 0, Rcurv, Rcurv)

# 2) cap‐box: x,y∈[−a,a], z∈[−s,0]
capbox_tag = gmsh.model.occ.addBox(-a, -a, s, 2*a, 2*a, -s)

gmsh.model.occ.synchronize()

# 3) spherical cap = intersection(sphere, cap_box)
topcap_dimtags_list, _ = gmsh.model.occ.intersect(
    [(3, sphere_tag)], [(3, capbox_tag)]
)
topcap_dimtag = topcap_dimtags_list[0]

# 4) cylindrical cap part: x,y∈[−a,a], z∈[−Lcap,−s]
capcyl_tag = gmsh.model.occ.addCylinder(
        0, 0, s,       # base center
        0, 0, Lcap - s,    # axis vector
        a                                # radius
    )
gmsh.model.occ.synchronize()

# 5) fuse top cap + cylindrical part
fuse_dimtags_list, _ = gmsh.model.occ.fuse([topcap_dimtag], [(3, capcyl_tag)])
cap_dimtag = fuse_dimtags_list[0]

gmsh.model.occ.synchronize()

z = Lcap
layers_tag = []
for h in layers_height:
    layers_tag.append(gmsh.model.occ.addCylinder(
            0, 0, z,     # base center
            0, 0, h,    # axis vector
            a            # radius
        ))
    z += h
gmsh.model.occ.synchronize()

if layers_tag:
    unfragmented_transducer_dimtags = [(3, tag) for tag in layers_tag] + [cap_dimtag]
    transducer_volumes_dimtags, transducer_volumes_mapping = gmsh.model.occ.fragment(
        unfragmented_transducer_dimtags,
        []
    )
else:
    transducer_volumes_dimtags = [cap_dimtag]
    transducer_volumes_mapping = {cap_dimtag[1]: cap_dimtag}
gmsh.model.occ.synchronize()

# 6) extract all surface faces of the fused volume
faces = gmsh.model.getBoundary(transducer_volumes_dimtags, oriented=False, recursive=False)
all_face_tags = [f[1] for f in faces]

print("Number of faces:", len(all_face_tags))
# 7) identify inlet: the face whose z_max ≈ 0
inlet_tag = None
tol = 1e-6
for tag in all_face_tags:
    # returns [xmin, ymin, zmin, xmax, ymax, zmax]
    bb = gmsh.model.getBoundingBox(2, tag)
    zmin, zmax = bb[2], bb[5]
    print(f"Face {tag}: zmin={zmin}, zmax={zmax}")
    if abs(zmin) < tol:
        inlet_tag = tag
        break
if inlet_tag is None:
    raise RuntimeError("Could not find inlet face at z≈0")

# 8) classify outflow = all other faces
outflow_tags = [tag for tag in all_face_tags if tag != inlet_tag]

# 9) physical groups & mesh
gmsh.model.addPhysicalGroup(2, [inlet_tag], inflow_marker)
gmsh.model.addPhysicalGroup(2, outflow_tags, outflow_marker)

# find all the layers of the same volumes tag
volumes_markers = {cap_volume_marker: [cap_dimtag[1]]}
for i, layer_tag in enumerate(layers_tag):
    # volume marker
    marker = layers_volume_markers[i] 
    if marker not in volumes_markers:
        volumes_markers[marker] = []
    volumes_markers[marker].append(layer_tag)


tag_counter = 1
fields = []
for marker, tags in volumes_markers.items():
    gmsh.model.addPhysicalGroup(3, tags, marker)
    
    fields += [tag_counter]
    gmsh.model.mesh.field.add("Constant", tag_counter)
    gmsh.model.mesh.field.setNumber(tag_counter, "VIn", materials_h[marker - 1])
    print(marker, tag_counter, materials_h[marker - 1] )
    gmsh.model.mesh.field.setNumbers(tag_counter, "VolumesList", tags)
    tag_counter += 1
    
gmsh.model.mesh.field.add("Min", tag_counter)
gmsh.model.mesh.field.setNumbers(tag_counter, "FieldsList", fields)
gmsh.model.mesh.field.setAsBackgroundMesh(tag_counter)

gmsh.model.mesh.generate(3)

# gmsh.fltk.run()

gmsh.write("data/transducer.msh")

gmsh.finalize()
