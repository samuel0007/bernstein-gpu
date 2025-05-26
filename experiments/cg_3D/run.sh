EXPERIMENT=cg_3D

# for p in 2 3 4 5 6 7 8 9; do
for p in 2; do
    python run.py mass3D_cg --scalar-type float64 --nvidia -p $p --rebuild 2>&1 | tee experiments/$EXPERIMENT/log_bernstein_$p.txt
done
