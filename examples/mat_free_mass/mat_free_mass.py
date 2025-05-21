# UFL input for the Matrix-free Poisson Demo
from basix.ufl import element
from basix import LagrangeVariant
from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    dx,
    inner,
)
import basix
import os


P = int(os.environ.get("polynomial_degree", 4))



coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

# Function Space
e = element("Lagrange", "triangle", P, lagrange_variant=LagrangeVariant.bernstein)
e_DG = element("Lagrange", "triangle", 0, discontinuous=True)

V = FunctionSpace(mesh, e)
V_DG = FunctionSpace(mesh, e_DG)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

alpha = Coefficient(V_DG)

# Bilinear and linear forms according to the variational
# formulation of the equations:
a = alpha * inner(u, v) * dx

# Linear form representing the action of the form `a`` on the
# coefficient `ui`:`
ui = Coefficient(V)
M = action(a, ui)

forms = [M]
