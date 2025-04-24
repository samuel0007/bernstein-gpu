EXPERIMENT=mat_free_RTX

nelements=256
for p in 2 3 4 5 6 7 8 9 10 11 12; do
    python run.py mat_free_mass --nvidia -p $p --rebuild --run-args --nelements $nelements --nreps 500 2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
done

# p=7
# python run.py mat_free_mass --nvidia -p $p --rebuild
# for nelements in 256; do
#     python run.py mat_free_mass --nvidia -p $p --run-args --nreps 200 --nelements $nelements 2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
# done