import basix

P = 2
element = basix.create_element(
    basix.ElementFamily.P,          
    basix.CellType.triangle, 
    P,                              
    basix.LagrangeVariant.bernstein, 
    basix.DPCVariant.unset, 
    False,                          
)


print(f"DOFs triangle {P}")
print(element.entity_dofs)

v0 = element.entity_dofs[0][0][0]
v1 = element.entity_dofs[0][1][0]
v2 = element.entity_dofs[0][2][0]

e0 = element.entity_dofs[1][0]
e1 = element.entity_dofs[1][1]
e2 = element.entity_dofs[1][2]  

c = element.entity_dofs[2][0]

print(f"v0: {v0}")
print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"e0: {e0}")
print(f"e1: {e1}")
print(f"e2: {e2}")
print(f"c: {c}")

c_count = 0
N = P + 1
dof_ordering = []
for i in range(N): # rows
    if(i == 0): # Bottom row
        dof_ordering += [v0]
        for i in range(P - 1):
            dof_ordering += [e2[i]]
        dof_ordering += [v1]
        continue
    if(i == N - 1): # Top row
        dof_ordering += [v2]
        continue
    
    # inner section
    dof_ordering += [e1[i - 1]]
    for j in range(P - 1 - i):
        dof_ordering += [c[c_count]]
        c_count += 1
    dof_ordering += [e0[i - 1]]
   
print(dof_ordering)
dof_ordering.reverse()
print(dof_ordering)

