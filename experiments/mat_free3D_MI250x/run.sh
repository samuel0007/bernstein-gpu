EXPERIMENT=mat_free3D_MI250x

nelements=10
for p in 3 4 5 6 7 8 9; do
    cd build
    cmake .. -Dpolynomial_degree=$p -DCMAKE_CXX_FLAGS="-I${MPICH_DIR}/include"   -DCMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
    make mat_free_mass3D
    cd ..
    srun --ntasks=1  build/mat_free_mass3D --nelements $nelements --nreps 500 2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
done