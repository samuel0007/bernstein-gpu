EXPERIMENT=stiff_cg
s=float64

# for p in 2 3 4 5 6; do
for p in 7; do
    cd build
    cmake .. -Dscalar_type=$s -Dpolynomial_degree=$p -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
    make stiff_cg
    cd ..
    build/stiff_cg - 2>&1 | tee experiments/$EXPERIMENT/log_bernstein_$p.txt
done