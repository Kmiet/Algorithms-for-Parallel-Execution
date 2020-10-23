#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --exclusive
#SBATCH --constraint="intel"
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-testing
#SBATCH --account=plgkrkmi2020a

POINTS=$1

loadMPI() {
  module add plgrid/tools/openmpi
  module add plgrid/tools/python
}

run() {
  mpirun -np 2 ./gauss seidel.py $POINTS
}

loadMPI &&
run