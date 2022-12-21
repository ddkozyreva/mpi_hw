#!/bin/sh
#SBATCH -J mpi_c
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -A proj_1339
#SBATCH --error=error.txt
#SBATCH --output=output.txt

mpicxx -O3 eigenvector.cpp -o main 
echo 1000 | srun --mpi=pmix ./main