EXPERIMENT=blockdim
s=float32
N=20


for p in 2 3 4 5 6 7 8; do
# for p in 4; do
    cd build
    cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
    make mat_free_stiffness3D
    cd ..
    for block_size in 16 24 32 45 64 74 92 122 128 177 256 512 729 1000 1024; do
        for cells_per_block in 1 2 3 4 5 6 7 8; do
            mpirun -n 1 build/mat_free_stiffness3D --nelements $N --block-size $block_size --cells-per-block $cells_per_block --nreps 100 - 2>&1 | tee experiments/$EXPERIMENT/log_stiffmem_${block_size}_${cells_per_block}_$p.txt
        done
    done
done

for p in 2 3 4 5 6 7 8; do
# for p in 4; do
    cd build
    cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
    make mat_free_stiffness3D
    cd ..
    for block_size in 16 24 32; do
        for cells_per_block in 9 10 11 12 13 14 15 16 32; do
            mpirun -n 1 build/mat_free_stiffness3D --nelements $N --block-size $block_size --cells-per-block $cells_per_block --nreps 100 - 2>&1 | tee experiments/$EXPERIMENT/log_stiffmem_${block_size}_${cells_per_block}_$p.txt
        done
    done
done

# for p in 8; do
# # for p in 4; do
#     cd build
#     cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
#     make mat_free_mass3D
#     cd ..
#     for block_size in 16 24 32 45 64 74 92 122 128 177 256 512 729 1000 1024; do
#         for cells_per_block in 1 2 3 4 5 6 7 8; do
#             mpirun -n 1 build/mat_free_mass3D --nelements $N --block-size $block_size --cells-per-block $cells_per_block --nreps 100 - 2>&1 | tee experiments/$EXPERIMENT/log_mass_${block_size}_${cells_per_block}_$p.txt
#         done
#     done
# done
