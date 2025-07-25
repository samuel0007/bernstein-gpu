#!/usr/bin/env bash
EXPERIMENT=sharedmem
s=float32
N=20

# per‑p optimal configs MASS
bs_for_p() {
    case "$1" in
        2)  echo 16 ;;
        3)  echo 24 ;;
        4)  echo 16 ;;
        5)  echo 16 ;;
        6)  echo 256 ;;
        7)  echo 16 ;;
        8)  echo 16 ;;
    esac
}
cpb_for_p() {
    case "$1" in
        2)  echo 8 ;;
        3)  echo 4 ;;
        4)  echo 2 ;;
        5)  echo 2 ;;
        6)  echo 1 ;;
        7)  echo 8 ;;
        8)  echo 8 ;;
    esac
}

# per‑p optimal configs STIFF
# bs_for_p() {
#     case "$1" in
#         2)  echo 16 ;;
#         3)  echo 24 ;;
#         4)  echo 16 ;;
#         5)  echo 16 ;;
#         6)  echo 16 ;;
#         7)  echo 16 ;;
#         8)  echo 16 ;;
#     esac
# }
# cpb_for_p() {
#     case "$1" in
#         2)  echo 2 ;;
#         3)  echo 4 ;;
#         4)  echo 32 ;;
#         5)  echo 15 ;;
#         6)  echo 16 ;;
#         7)  echo 4 ;;
#         8)  echo 2 ;;
#     esac
# }


mkdir -p "experiments/$EXPERIMENT"

for p in 2 3 4 5 6 7 8; do
# for p in 4; do
    cd build
    cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
    make mat_free_mass3D -j"$(nproc)"
    cd ..

    bs=$(bs_for_p "$p")
    cpb=$(cpb_for_p "$p")

    # for N in 5 10 15 20 25 30; do
        # for jacobian_type in 1 2 3 4; do
    mpirun -n 1 build/mat_free_mass3D \
        --nelements "$N" \
        --nreps 100 \
        --block-size "$bs" \
        --cells-per-block "$cpb" \
        2>&1 | tee "experiments/$EXPERIMENT/log_massT_$p.txt"
        # done
    # done
done
