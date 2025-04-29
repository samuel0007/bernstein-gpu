import basix

P = 4
element = basix.create_element(
    basix.ElementFamily.P,          
    basix.CellType.interval, 
    P,                              
    basix.LagrangeVariant.bernstein, 
    basix.DPCVariant.unset, 
    False,                          
)


print(f"DOFs interval {P}")
print(element.entity_dofs)

v0 = element.entity_dofs[0][0][0]
v1 = element.entity_dofs[0][1][0]
e0 = element.entity_dofs[1][0]

# dof ordering is v0, e0, v1
dof_ordering = [v0] + e0 + [v1]
print(dof_ordering)

# print(dof_ordering)
# dof_ordering.reverse()
# print(dof_ordering)

