EXPERIMENT=KO_RTX
s=float32

for p in 2 3 4 5 6 7 8; do
# for p in 4; do
    cd build
    cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
    make newmark_cg 2>&1 | tee ../experiments/$EXPERIMENT/log_lb_newmark_cg_$p.txt 2>&1
    cd ..
    # for N in 5 10 15 20 25 30; do
        # for jacobian_type in 1 2 3 4; do
            # mpirun -n 1 build/newmark_cg --N $N --jacobian-type $jacobian_type - 2>&1 | tee experiments/$EXPERIMENT/log_${jacobian_type}_${N}_$p.txt
        # done
    # done
done
