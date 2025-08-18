# This scripts runs all the four tUS benchmark problems at P4
P=4
s=float64 # float64
sample_nx=251 # Samples for FFT in the x direction
sample_nz=251 # Samples for FFT in the z direction
cg_tol=1e-3 # Relative tolerance of conjugate gradient solver
CFL=1.0
source_frequency=0.5e6
sample_harmonic=20
sample_periods=4
buffer_periods=8

validation_folder=/home/sr2120/benchmark_data/KWAVE
paraview_python=../rendering/ParaView-5.13.2-egl-MPI-Linux-Python3.10-x86_64/bin/pvpython

spack env activate mfenenv
spack load cmake

cd build
cmake .. -Dscalar_type=$s -Dpolynomial_degree=$P -Dnvidia=On -Damd=off -DCMAKE_BUILD_TYPE=Release
cmake --build . --target FreeFUS

./apps/FreeFUS/FreeFUS --mesh BP1 --material-case 1 --source-frequency ${source_frequency} --cg-tol ${cg_tol} --CFL ${CFL} --sample-harmonic ${sample_harmonic} --sample-periods ${sample_periods} --buffer-periods ${buffer_periods} --insitu-output-steps 1000 --output-steps 1000 --output-path output-BP1 | tee BP1.log
# ./apps/FreeFUS/FreeFUS --mesh BP2 --material-case 2 --source-frequency ${source_frequency} --cg-tol ${cg_tol} --CFL ${CFL} --sample-harmonic ${sample_harmonic} --sample-periods ${sample_periods} --buffer-periods ${buffer_periods} --insitu-output-steps 1000 --output-steps 1000 --output-path output-BP2 | tee BP2.log
# ./apps/FreeFUS/FreeFUS --mesh BP3 --material-case 3 --source-frequency ${source_frequency} --cg-tol ${cg_tol} --CFL ${CFL} --sample-harmonic ${sample_harmonic} --sample-periods ${sample_periods} --buffer-periods ${buffer_periods} --insitu-output-steps 1000 --output-steps 1000 --output-path output-BP3 | tee BP3.log
# ./apps/FreeFUS/FreeFUS --mesh BP4 --material-case 4 --source-frequency ${source_frequency} --cg-tol ${cg_tol} --CFL ${CFL} --sample-harmonic ${sample_harmonic} --sample-periods ${sample_periods} --buffer-periods ${buffer_periods} --insitu-output-steps 1000 --output-steps 1000 --output-path output-BP4 | tee BP4.log

# analysis
cd ../analysis

spack env deactivate # pvpython really dislikes being in a spack env
. ../.env/bin/activate

../$paraview_python sliced_pvplot.py ../apps/FreeFUS/data/BP1/output-BP1-sliced.bp $((sample_harmonic*sample_periods)) --freq $source_frequency --dims $sample_nx 1 $sample_nz --output_dir ./output-BP1
# ../$paraview_python sliced_pvplot.py ../apps/FreeFUS/data/BP2/output-BP2-sliced.bp $((sample_harmonic*sample_periods)) --freq $source_frequency --dims $sample_nx 1 $sample_nz --output_dir ./output-BP2
# ../$paraview_python sliced_pvplot.py ../apps/FreeFUS/data/BP3/output-BP3-sliced.bp $((sample_harmonic*sample_periods)) --freq $source_frequency --dims $sample_nx 1 $sample_nz --output_dir ./output-BP3
# ../$paraview_python sliced_pvplot.py ../apps/FreeFUS/data/BP4/output-BP4-sliced.bp $((sample_harmonic*sample_periods)) --freq $source_frequency --dims $sample_nx 1 $sample_nz --output_dir ./output-BP4

python validation.py --comparison-folder output-BP1 --source 1 --bm 1 --validation-folder ${validation_folder} | tee BP1_validation.log
# python validation.py --comparison-folder output-BP2 --source 1 --bm 2 --validation-folder ${validation_folder} | tee BP2_validation.log
# python validation.py --comparison-folder output-BP3 --source 1 --bm 3 --validation-folder ${validation_folder} | tee BP3_validation.log
# python validation.py --comparison-folder output-BP4 --source 1 --bm 4 --validation-folder ${validation_folder} | tee BP4_validation.log