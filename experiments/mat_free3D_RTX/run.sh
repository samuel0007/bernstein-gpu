EXPERIMENT=mat_free3D_RTX

nelements=10
for p in 3 4 5 6 7 8 9 10 11 12; do
    python run.py mat_free_mass3D --nvidia -p $p --rebuild --run-args --nelements $nelements --nreps 200 2>&1 | tee experiments/$EXPERIMENT/bp_log_$p.txt
done

# p=7   
# python run.py mat_free_mass --nvidia -p $p --rebuild
# for nelements in 256; do
#     python run.py mat_free_mass --nvidia -p $p --run-args --nreps 200 --nelements $nelements 2>&1 | tee experiments/$EXPERIMENT/log_$p.txt
# done