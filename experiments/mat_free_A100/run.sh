EXPERIMENT=mat_free_A100
nelements=256

module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load gcc/11.3.0/gcc/i4xnp7h5
module load openmpi/4.1.4/gcc/6kaxusn4
spack load cuda
spack load cmake

# float 32
for p in 2 3 4 5 6 7 8 9; do
    python run.py mat_free_mass --scalar-type float32 --nvidia -p $p --rebuild --run-args --nelements $nelements --nreps 500 2>&1 | tee experiments/$EXPERIMENT/log_32s_$p.txt
done

# float 64
for p in 2 3 4 5 6 7 8 9; do
    python run.py mat_free_mass --scalar-type float64 --nvidia -p $p --rebuild --run-args --nelements $nelements --nreps 500 2>&1 | tee experiments/$EXPERIMENT/log_64s_$p.txt
done