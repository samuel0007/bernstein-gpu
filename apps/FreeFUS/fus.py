import basix
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Mesh, TrialFunction, TestFunction,
                 ds, dx, grad, inner, action)
import os

lagrange_variants_map = {
    "gll_warped": basix.LagrangeVariant.gll_warped,
    "equispaced": basix.LagrangeVariant.equispaced,
    "bernstein": basix.LagrangeVariant.berstein,
}

P = int(os.environ.get("polynomial_degree", 4))
LVARIANT = os.environ.get("LVARIANT", "gll_warped")
DIMENSION = int(os.environ.get("DIMENSION", 3))

cell_type = basix.CellType.tetrahedron if DIMENSION == 3 else basix.CellType.triangle
cell_type_name = "tetrahedron" if DIMENSION == 3 else "triangle"
lagrange_variant = lagrange_variants_map[LVARIANT]

coord_element = element("Lagrange", cell_type_name, 1, shape=(3, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, cell_type, P,
    lagrange_variant)
e_DG = element(basix.ElementFamily.P, cell_type, 0,
    lagrange_variant, basix.DPCVariant.unset, True)

# Define function spaces
V = FunctionSpace(mesh, e)
V_DG = FunctionSpace(mesh, e_DG)

c0 = Coefficient(V_DG)
rho0 = Coefficient(V_DG)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
g = Coefficient(V)
v = TestFunction(V)

forms = []
# Linear Wave equation
a_linearwave = inner(u/rho0/c0/c0, v) * dx
L_linearwave = - inner(1/rho0*grad(u_n), grad(v)) * dx \
    + inner(1/rho0*g, v) * ds(1) \
    - inner(1/rho0/c0*v_n, v) * ds(2)

    
# Lossy Linear Wave equation

# Westervelts Equation

forms = [a_linearwave, L_linearwave]
