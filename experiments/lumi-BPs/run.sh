#!/bin/bash
#SBATCH --job-name=BPs
#SBATCH --account=project_465001872
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8 
#SBATCH --gpus-per-node=8
#SBATCH --partition=dev-g

# ~20 minutes per simulation at p6 with CFL 10.0
cd /scratch/project_465001872/russosam
. spack/share/spack/setup-env.sh
spack env activate adios2-hypre-rocm

export MPICH_GPU_SUPPORT_ENABLED=1

# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP1 --material-case 1 --CFL 10.0 --source-frequency 500000 --output-steps 100000 --insitu-output-steps 100000
# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP2 --material-case 2 --CFL 10.0 --source-frequency 500000 --output-steps 100000 --insitu-output-steps 100000
# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP3 --material-case 3 --CFL 10.0 --source-frequency 500000 --output-steps 100000 --insitu-output-steps 100000
# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP4 --material-case 4 --CFL 10.0 --source-frequency 500000 --output-steps 100000 --insitu-output-steps 100000
srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh skull-medium --material-case 9 --CFL 10.0 --source-frequency 500000 --output-steps 100000 --insitu-output-steps 100000

