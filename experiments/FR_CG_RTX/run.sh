#!/usr/bin/env bash
EXPERIMENT=FR_CG_RTX
s=float32

bs_for_p() {
    case "$1" in
        2)  echo 16 ;;
        3)  echo 24 ;;
        4)  echo 45 ;;
        5)  echo 16 ;;
        6)  echo 64 ;;
        7)  echo 16 ;;
        8)  echo 32 ;;
    esac
}
cpb_for_p() {
    case "$1" in
        2)  echo 2 ;;
        3)  echo 4 ;;
        4)  echo 3 ;;
        5)  echo 16 ;;
        6)  echo 8 ;;
        7)  echo 4 ;;
        8)  echo 2 ;;
    esac
}

mkdir -p "experiments/$EXPERIMENT"

# for p in 2 3 4 5 6 7 8; do
for p in 4; do
    cd build
    cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
    make newmark_cg -j"$(nproc)"
    cd ..

    bs=$(bs_for_p "$p")
    cpb=$(cpb_for_p "$p")

    for N in 5 10 15 20 25 30; do
        for jacobian_type in 1 2 3 4; do
            mpirun -n 1 build/newmark_cg \
                --N "$N" \
                --jacobian-type "$jacobian_type" \
                --block-size "$bs" \
                --cells-per-block "$cpb" \
                2>&1 | tee "experiments/$EXPERIMENT/log_${jacobian_type}_${N}_$p.txt"
        done
    done
done
