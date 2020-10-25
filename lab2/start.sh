#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 5
#SBATCH --constraint="intel"
#SBATCH --time=00:05:00
#SBATCH --partition=plgrid-testing
#SBATCH --account=plgkrkmi2020a

N=$1
M=$2
G=$3
L=$4
E=$5

loadMPI() {
  module add plgrid/tools/openmpi
  module add plgrid/tools/python-intel/3.6.5
}

run() {
  mpirun -np 5 ./gauss_seidel.py $N $M $G $L $E
}

loadMPI &&
run