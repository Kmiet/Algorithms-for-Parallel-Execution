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
  module add plgrid/tools/python-intel/3.6.5
}

run() {
  mpirun -np 2 ./erastotenes.py $POINTS
}

loadMPI &&
run