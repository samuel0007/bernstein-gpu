EXPERIMENT=mat_free_RTX

# for p in range (1, 6):
for p in 1 2 3 4 5; do
    python run.py mat_free_mass --nvidia -p $p --rebuild --run-args --nreps 200 2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
done