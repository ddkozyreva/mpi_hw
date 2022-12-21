#!/bin/sh
#SBATCH --job-name=hw3
#SBATCH --account=proj_1339        # Название проекта (курса)
#SBATCH --ntasks=1                 # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=32         # Требуемое кол-во CPU
#SBATCH --constraint="type_a|type_b|type_c|type_d"
#SBATCH --time=2


mpicxx -O3 eigenvector.cpp -o main 
echo 1000 | mpirun ./main  > output.txt
