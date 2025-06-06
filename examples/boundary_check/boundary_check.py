# UFL input for the Matrix-free Poisson Demo
from basix.ufl import element
from basix import LagrangeVariant
from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    ds,
    inner,
)
import os


P = int(os.environ.get("polynomial_degree", 4))



coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

# Function Space
e = element("Lagrange", "triangle", P, lagrange_variant=LagrangeVariant.bernstein)
e_DG = element("Lagrange", "triangle", 0, discontinuous=True)

V = FunctionSpace(mesh, e)
V_DG = FunctionSpace(mesh, e_DG) # for coefficients

# Trial and test functions
g = Coefficient(V)
v = TestFunction(V)



L = inner(g, v) * ds


forms = [L]
