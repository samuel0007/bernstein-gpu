EXPERIMENT=mat_free3D_A30

nelements=10
# for p in 1 2 3 4 5 6 7 8 9; do
for p in 2 3 4 5 6 7 8 9; do
    python run.py mat_free_mass3D --nvidia -p $p --rebuild --run-args --nelements $nelements --nreps 100 2>&1 | tee experiments/$EXPERIMENT/sf_log_$p.txt
done
