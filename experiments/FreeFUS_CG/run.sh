EXPERIMENT=FreeFUS_CG
s=float32
max_steps=100
sample_harmonic=0

bs_for_p() {
    case "$1" in
        2)  echo 16 ;;
        3)  echo 24 ;;
        4)  echo 16 ;;
        5)  echo 16 ;;
        6)  echo 16 ;;
        7)  echo 16 ;;
        8)  echo 16 ;;
    esac
}
cpb_for_p() {
    case "$1" in
        2)  echo 2 ;;
        3)  echo 4 ;;
        4)  echo 32 ;;
        5)  echo 15 ;;
        6)  echo 16 ;;
        7)  echo 4 ;;
        8)  echo 2 ;;
    esac
}

for p in 2 3 4 5 6 7 8 9; do
# for p in 10 11 12 13 14; do
        cd build/apps/FreeFUS
        cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off
        make FreeFUS
        cd ../../..
    bs=$(bs_for_p "$p")
    cpb=$(cpb_for_p "$p")

    for CFL in 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 100.0; do
        mpirun -n 1 build/apps/FreeFUS --CFL $CFL --sample-harmonic $sample_harmonic --max-steps $max_steps  2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
    done
done
