EXPERIMENT=mat_free_RTX

p=7
nelements=256

# for p in range (1, 6):
# for p in 1 2 3 4 5; do
#     python run.py mat_free_mass --nvidia -p $p --rebuild --run-args --nreps 200 2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
# done
# python run.py mat_free_mass --nvidia -p $p --rebuild

for nelements in 256; do
    python run.py mat_free_mass --nvidia -p $p --run-args --nreps 200 --nelements $nelements 2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
done