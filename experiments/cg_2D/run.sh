EXPERIMENT=cg_2D

for p in 2 3 4 5 6 7 8 9 10 11 12; do
    python run.py mass_cg --scalar-type float64 --nvidia -p $p --rebuild 2>&1 | tee experiments/$EXPERIMENT/log_gll_$p.txt
done
