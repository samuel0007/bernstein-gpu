#!/bin/bash
#SBATCH --job-name=BP4
#SBATCH --account=project_465001872
#SBATCH --time=01:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8 
#SBATCH --gpus-per-node=8
#SBATCH --partition=standard-g

# ~30 minutes per simulation at p6 with CFL 6.0 with 8 nodes
cd /scratch/project_465001872/russosam
. spack/share/spack/setup-env.sh
spack env activate adios2-hypre-rocm

export MPICH_GPU_SUPPORT_ENABLED=1

# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP1 --material-case 1 --CFL 6.0 --source-frequency 500000 --sample-nz 1000 --sample-harmonic 2 --output-steps 1000000 --insitu-output-steps 1000000
# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP2 --material-case 2 --CFL 6.0 --source-frequency 500000 --sample-nz 1000 --sample-harmonic 2 --output-steps 1000000 --insitu-output-steps 1000000
# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP3 --material-case 3 --CFL 6.0 --source-frequency 500000 --sample-nz 1000 --sample-harmonic 2 --output-steps 1000000 --insitu-output-steps 1000000
srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh BP4 --material-case 4 --CFL 6.0 --source-frequency 500000 --sample-nz 1000 --sample-harmonic 2 --output-steps 1000000 --insitu-output-steps 1000000
# srun /scratch/project_465001872/russosam/bernstein-gpu/build/apps/FreeFUS/FreeFUS --mesh skull-medium --material-case 9 --CFL 6.0 --source-frequency 300000 --sample-nz 1000 --sample-harmonic 2 --output-steps 1000000 --insitu-output-steps 1000000

