EXPERIMENT=mat_free_A100
nelements=256

module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load gcc/11.3.0/gcc/i4xnp7h5
module load openmpi/4.1.4/gcc/6kaxusn4
spack load cuda
spack load cmake

# float 32
for s in float32 float64; do
    for p in 2 3 4 5 6 7 8 9; do
        cd build
        cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off
        make mat_free_mass
        cd ..
        mpirun -n 1 build/mat_free_mass --nreps 500 --nelements 256 2>&1 | tee experiments/$EXPERIMENT/log_${s}_$p.txt
    done
done


