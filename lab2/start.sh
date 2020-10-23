#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --exclusive
#SBATCH --constraint="intel"
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-testing
#SBATCH --account=plgkrkmi2020a

N=$1
M=$2
G=$3
L=$4
E=$5

# loadMPI() {
#   module add plgrid/tools/openmpi
#   module add plgrid/tools/python
# }

run() {
  mpirun -np 5 ./gauss_seidel.py $N $M $G $L $E
}

# loadMPI &&
run