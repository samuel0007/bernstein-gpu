import meshio
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
import numpy as np

import dolfinx
import ufl
import basix
import gmsh

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh

def get_cell_data(mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    markers = mesh.get_cell_data("gmsh:physical", cell_type)
    return cells, mesh.points, markers


fname = "medium_skull.msh"
gmsh.initialize()
gmsh.merge(fname)
gmsh.model.geo.remove_all_duplicates()
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("skull_mesh_duplicate.msh")


mesh_from_file = meshio.read("skull_mesh_duplicate.msh")
cells, points, markers = get_cell_data(mesh_from_file, "tetra")

print(markers)
print("Cells shape:", cells.shape)
print("Points shape:", points.shape)
print("Markers shape:", markers.shape)

msh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    cells.astype(np.int64),
    points,
    ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
)


local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
    msh,
    msh.topology.dim,
    cells.astype(np.int64),
    markers.astype(np.int32, copy=False),
)

adj = dolfinx.graph.adjacencylist(local_entities)
ct = dolfinx.mesh.meshtags_from_entities(
    msh,
    msh.topology.dim,
    adj,
    local_values.astype(np.int32, copy=False),
)
with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='mesh']/Geometry")
