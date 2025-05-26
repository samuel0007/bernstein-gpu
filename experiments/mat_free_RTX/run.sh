EXPERIMENT=mat_free_RTX

nelements=256
for s in float32 float64; do
    for p in 2 3 4 5 6 7 8 9; do
        cd build
        cmake .. -Dscalar_type=${s} -Dpolynomial_degree=$p -Dnvidia=On -Damd=off
        make mat_free_mass
        cd ..
	    mpirun -n 1 build/mat_free_mass --nreps 500 --nelements $nelements 2>&1 | tee experiments/$EXPERIMENT/log_${s}_$p.txt
    done
done
