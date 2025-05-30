import pygmsh
import math
import gmsh

filename = "data/transducer.msh"
# TYPE = "spherical"
# TYPE = "planar"
TYPE = "simple"

D = 3
# R = 0.064 if TYPE == "spherical" else 0.01 # [m]
R = 0.064 if TYPE == "spherical" else 0.035 # [m]
# R = 0.016 if TYPE == "spherical" else 0.005 # [m]

A = 0.064 # [m]
# A = 0.016 # [m]
H = 0.02
# H = 0.120 # [m]
# H = 0.015 # [m]
# W = 0.070 # [m]
W = 0.01
# W = 0.012 # [m]

frequency = 0.5e6 # [Hz]
speedOfSound = 1500 # [m/s]
wavelength = speedOfSound / frequency # [m]
elementsPerWavelength = 4 # [elements]
inflow_marker = 1
outflow_marker = 2
volume_marker = 1

h = wavelength / elementsPerWavelength


def add_spherical_transducer_2d(model, radius_of_curvature, aperture_diameter, box_length, h):
    a = aperture_diameter / 2.0
    s = radius_of_curvature - math.sqrt(radius_of_curvature**2 - a**2)

    # points for circular arc
    p_left   = model.add_point([-a,   0.0, 0.0], mesh_size=h)
    p_center = model.add_point([ 0.0, -s,   0.0], mesh_size=h)
    p_right  = model.add_point([ a,   0.0, 0.0], mesh_size=h)
    arc      = model.add_circle_arc(p_left, p_center, p_right)

    # box points
    p_br = model.add_point([ a, -box_length, 0.0], mesh_size=h)
    p_bl = model.add_point([-a, -box_length, 0.0], mesh_size=h)
    l1   = model.add_line(p_right, p_br)
    l2   = model.add_line(p_br,    p_bl)
    l3   = model.add_line(p_bl,    p_left)

    outflow_b = [l1, l2, l3]
    boundary = model.add_curve_loop([arc, *outflow_b])
    surface = model.add_plane_surface(boundary)
    return arc, outflow_b, [surface]

def add_simple_transducer_2d(model, box_length, box_width, h):
    # points for the rectangular transducer
    p_tl = model.add_point([-box_width/2.0, 0.0, 0.0], mesh_size=h)
    p_tr = model.add_point([ box_width/2.0, 0.0, 0.0], mesh_size=h)
    p_bl = model.add_point([-box_width/2.0, -box_length, 0.0], mesh_size=h)
    p_br = model.add_point([ box_width/2.0, -box_length, 0.0], mesh_size=h)

    # lines for the rectangular transducer
    l_r = model.add_line(p_tr, p_br)
    l_b = model.add_line(p_br, p_bl)
    l_l = model.add_line(p_bl, p_tl)
    l_t = model.add_line(p_tl, p_tr)
    
    outflow = [l_b]
    inflow = l_t
    
    boundary = model.add_curve_loop([l_t, l_r, l_b, l_l])
    surface  = model.add_plane_surface(boundary)
    
    return inflow, outflow, [surface]

def add_simple_transducer_3d(model, box_length, box_width, h):
    # corners
    x0, x1 = -box_width/2.0, box_width/2.0
    y0, y1 =    0.0       , -box_length
    z0, z1 =    0.0       ,  box_width
    p = lambda x,y,z: model.add_point([x,y,z], mesh_size=h)
    p000 = p(x0,y0,z0); p100 = p(x1,y0,z0)
    p110 = p(x1,y1,z0); p010 = p(x0,y1,z0)
    p001 = p(x0,y0,z1); p101 = p(x1,y0,z1)
    p111 = p(x1,y1,z1); p011 = p(x0,y1,z1)

    # edges
    L = lambda a,b: model.add_line(a,b)
    l1 = L(p000,p100);  l2 = L(p100,p110)
    l3 = L(p110,p010);  l4 = L(p010,p000)
    l5 = L(p001,p101);  l6 = L(p101,p111)
    l7 = L(p111,p011);  l8 = L(p011,p001)
    l9 = L(p000,p001); l10 = L(p100,p101)
    l11= L(p110,p111); l12 = L(p010,p011)

    # faces (curve loops + plane surfaces)
    front_loop  = model.add_curve_loop([l1,  l10, -l5,  -l9])
    back_loop   = model.add_curve_loop([l3,  l12, -l7,  -l11])
    left_loop   = model.add_curve_loop([l4,   l9, -l8,  -l12])
    right_loop  = model.add_curve_loop([l2,  l11, -l6,  -l10])
    bottom_loop = model.add_curve_loop([l1,   l2,  l3,   l4])
    top_loop    = model.add_curve_loop([l5,   l6,  l7,   l8])

    front  = model.add_plane_surface(front_loop)
    back   = model.add_plane_surface(back_loop)
    left   = model.add_plane_surface(left_loop)
    right  = model.add_plane_surface(right_loop)
    bottom = model.add_plane_surface(bottom_loop)
    top    = model.add_plane_surface(top_loop)

    # volume
    sl = model.add_surface_loop([front, right, back, left, bottom, top])
    vol = model.add_volume(sl)

    inflow, outflow = front, [back]
    return inflow, outflow, [vol]
    
    
def add_planar_transducer_2d(model, radius, box_length, box_width, h):
    a = radius
    w = box_width / 2.0

    # top points
    p_tl    = model.add_point([-w,  0.0,     0.0], mesh_size=h)
    p_il    = model.add_point([-a,  0.0,     0.0], mesh_size=h)
    p_ir    = model.add_point([ a,  0.0,     0.0], mesh_size=h)
    p_tr    = model.add_point([ w,  0.0,     0.0], mesh_size=h)

    # bottom points
    p_br    = model.add_point([ w, -box_length, 0.0], mesh_size=h)
    p_bl    = model.add_point([-w, -box_length, 0.0], mesh_size=h)

    # top boundary: left segment, inflow segment, right segment
    l_tl    = model.add_line(p_tl, p_il)
    inflow  = model.add_line(p_il, p_ir)
    l_tr    = model.add_line(p_ir, p_tr)

    # side and bottom
    l_r     = model.add_line(p_tr, p_br)
    l_b     = model.add_line(p_br, p_bl)
    l_l     = model.add_line(p_bl, p_tl)

    outflow = [l_tl, l_tr, l_r, l_b, l_l]

    boundary = model.add_curve_loop([l_tl, inflow, l_tr, l_r, l_b, l_l])
    surface  = model.add_plane_surface(boundary)

    return inflow, outflow, [surface]

# def add_spherical_transducer_3d(radius_of_curvature, aperture_diameter, box_length, mesh_size):
#     a = aperture_diameter/2.0
#     s = radius_of_curvature - math.sqrt(radius_of_curvature**2 - a**2)

#     # 1) full sphere so its cap at z=0
#     sphere_tag = gmsh.model.occ.addSphere(0, 0, -radius_of_curvature, radius_of_curvature)
#     # 2) cap‐box: x,y∈[−a,a], z∈[−s,0]
#     cap_box_tag = gmsh.model.occ.addBox(-a, -a, -s, 2*a, 2*a, s)
#     gmsh.model.occ.synchronize()

#     # 3) spherical cap = sphere ∩ cap_box
#     cap_list, _ = gmsh.model.occ.intersect([(3, sphere_tag)], [(3, cap_box_tag)])
#     cap = cap_list[0]

#     # 4) backing cylinder: radius=a, z∈[−box_length,−s]
#     back_cyl = gmsh.model.occ.addCylinder(0, 0, -box_length, 0, 0, box_length - s, a)
#     gmsh.model.occ.synchronize()

#     # 5) fuse cap + backing
#     fuse_list, _ = gmsh.model.occ.fuse([cap], [(3, back_cyl)])
#     fused = fuse_list[0]
#     gmsh.model.occ.synchronize()

#     # 6) extract all surface faces
#     faces = gmsh.model.getBoundary([(3, fused[1])], oriented=False, recursive=False)
#     tags = [f[1] for f in faces]

#     # 7) find inlet (z_max≈0)
#     inlet = None
#     tol = 1e-6
#     for t in tags:
#         _,_,zmin,_,_,zmax = gmsh.model.getBoundingBox(2, t)
#         if abs(zmax) < tol and zmin < tol:
#             inlet = t
#             break
#     if inlet is None:
#         raise RuntimeError("Inlet face not found")

#     # 8) outflow = all other faces
#     outflow = [t for t in tags if t != inlet]

#     for dim, tag in gmsh.model.getEntities(0):
#         gmsh.model.mesh.setSize([(dim, tag)], mesh_size)
#     return inlet, outflow, fused[1]



def add_planar_transducer_3d(model, radius, box_length, box_width, h):
    R = box_width/2.0

    # --- top circle (domain boundary) ---
    p0o = model.add_point([ R, 0.0, 0.0], mesh_size=h)
    p1o = model.add_point([ 0.0, R,  0.0], mesh_size=h)
    p2o = model.add_point([-R, 0.0, 0.0], mesh_size=h)
    p3o = model.add_point([ 0.0,-R,  0.0], mesh_size=h)
    pco = model.add_point([ 0.0, 0.0, 0.0], mesh_size=h)

    a0o = model.add_circle_arc(p0o, pco, p1o)
    a1o = model.add_circle_arc(p1o, pco, p2o)
    a2o = model.add_circle_arc(p2o, pco, p3o)
    a3o = model.add_circle_arc(p3o, pco, p0o)

    outer_loop = model.add_curve_loop([a0o, a1o, a2o, a3o])
    # outer_surf = model.add_plane_surface(outer_loop)

    # --- inlet circle ---
    p0i = model.add_point([ radius,  0.0, 0.0], mesh_size=h)
    p1i = model.add_point([ 0.0,  radius, 0.0], mesh_size=h)
    p2i = model.add_point([-radius,  0.0, 0.0], mesh_size=h)
    p3i = model.add_point([ 0.0, -radius, 0.0], mesh_size=h)
    pci = model.add_point([ 0.0,    0.0, 0.0], mesh_size=h)

    a0i = model.add_circle_arc(p0i, pci, p1i)
    a1i = model.add_circle_arc(p1i, pci, p2i)
    a2i = model.add_circle_arc(p2i, pci, p3i)
    a3i = model.add_circle_arc(p3i, pci, p0i)

    inlet_loop = model.add_curve_loop([a0i, a1i, a2i, a3i])
    inlet_surf = model.add_plane_surface(inlet_loop)

    model.synchronize()
    
    annulus_surf = model.add_plane_surface(outer_loop, [inlet_loop])


    # --- extrude annulus ⇒ hollow cylinder volume + its lateral walls + top ring ---
    top_annulus, vol_annulus, lateral_annulus = model.extrude(
        annulus_surf,
        [0, 0, -box_length]
    )

    # --- extrude inlet disk ⇒ small cylinder (transducer) + its lateral walls + top cap ---
    top_inlet, vol_inlet, lateral_inlet = model.extrude(
        inlet_surf,
        [0, 0, -box_length]
    )

    # collect surfaces:
    # outflow_surfs = [top_annulus, annulus_surf, top_inlet] + lateral_annulus[:4]
    outflow_surfs = [top_annulus, top_inlet] + lateral_annulus[:4]
    inlet_surfs   = inlet_surf

    # return lists of tags (you can assign physical groups thereafter)
    return inlet_surfs, outflow_surfs, [vol_annulus, vol_inlet]



print("Generating mesh:")
print("\tSpeed of sound: ", speedOfSound)
print("\tRadius of curvature: ", R)
print("\tAperture diameter: ", A)
print("\tBox length: ", H)
print("\tElements per wavelength: ", elementsPerWavelength)
print("\tFrequency: ", frequency)
print("\tWavelength: ", wavelength)
print("\tMesh size: ", h)


with pygmsh.occ.Geometry() as geom:
    if TYPE == "spherical":
        if D == 2:
            inflow_b, outflow_b, surface = add_spherical_transducer_2d(geom, R, A, H, h)
    elif TYPE == "planar":
        if D == 2:
            inflow_b, outflow_b, surface = add_planar_transducer_2d(geom, R, H, W, h)
        else:
            inflow_b, outflow_b, surface = add_planar_transducer_3d(geom, R, H, W, h)
    elif TYPE == "simple":
        if D == 2:
            inflow_b, outflow_b, surface = add_simple_transducer_2d(geom, H, W, h)
        else:
            inflow_b, outflow_b, surface = add_simple_transducer_3d(geom, H, W, h)
    else:
        raise ValueError("Invalid transducer type. Choose 'spherical' or 'planar'.")
    geom.synchronize()
    gmsh.model.addPhysicalGroup(D - 1, [inflow_b.dim_tags[0][1]], inflow_marker)
    # gmsh.model.setPhysicalName(1, inflow_marker, "Inflow")
    gmsh.model.addPhysicalGroup(D - 1, [outflow_b[i].dim_tags[0][1] for i in range(len(outflow_b))], outflow_marker)
    # gmsh.model.setPhysicalName(1, outflow_marker, "Outflow")
    gmsh.model.addPhysicalGroup(D, [surface[i].dim_tags[0][1] for i in range(len(surface))], volume_marker)
    # gmsh.model.setPhysicalName(2, volume_marker, "Volume")
    gmsh.model.mesh.generate(D)
    
    gmsh.write(filename)
