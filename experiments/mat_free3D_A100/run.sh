EXPERIMENT=mat_free3D_A100
nelements=10

module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load gcc/11.3.0/gcc/i4xnp7h5
module load openmpi/4.1.4/gcc/6kaxusn4
spack load cuda
spack load cmake

for s in float32 float64; do                                                                                                                                                                                                                             # float 32                                                                                                                  for s in float32 float64; do
    for p in 2 3 4 5 6 7 8 9; do
        cd build
        cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off
        make mat_free_mass3D
        cd ..
        mpirun -n 1 build/mat_free_mass3D --nreps 500 --nelements 256 2>&1 | tee experiments/$EXPERIMENT/log_$s_$p.txt
    done
done