import matplotlib.pyplot as plt
import numpy as np
from basix.ufl import element
from basix import LagrangeVariant
import basix

e = basix.create_element(
    basix.ElementFamily.P,          
    basix.CellType.triangle, 
    2,                              
    basix.LagrangeVariant.bernstein, 
    basix.DPCVariant.unset, 
    False,                          
)


print(e.entity_dofs)
print(e.dof_ordering)