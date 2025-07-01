EXPERIMENT=mat_free_MI250x

module load LUMI/23.09
module load partition/G
module load rocm
module load gcc
module load craype
module load cray-mpich
module load craype-accel-amd-gfx90a
spack load cmake


export CC=gcc-12
export CXX=g++-12


nelements=256
for s in float32 float64; do
    for p in 2 3 4 5 6 7 8 9; do
        cd build
        cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=Off -Damd=On -DCMAKE_CXX_FLAGS="-I${MPICH_DIR}/include"   -DCMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
        make mat_free_mass
        cd ..
	srun --ntasks=1 build/mat_free_mass --nreps 500 --nelements $nelements 2>&1 | tee experiments/$EXPERIMENT/log_${s}_$p.txt
    done
done
