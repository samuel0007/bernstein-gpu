import basix

for p in range(1, 6):
    element = basix.create_element(
        basix.ElementFamily.P,          
        basix.CellType.tetrahedron, 
        p,                              
        basix.LagrangeVariant.bernstein, 
        basix.DPCVariant.unset, 
        False,                          
    )
    print(f"p: {p}, Identity transform: {element.dof_transformations_are_identity}")
    print(f"p: {p}, Permutation transform: {element.dof_transformations_are_permutations}")


