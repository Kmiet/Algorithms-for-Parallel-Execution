!/bin/bash -l
SBATCH --ntasks 32
SBATCH --constraint="intel"
SBATCH --time=00:05:00
SBATCH --partition=plgrid-testing
SBATCH --account=plgkrkmi2020a

N=$1

loadMPI() {
  module add plgrid/tools/openmpi
  module add plgrid/tools/python-intel/3.6.5
}

run() {
  mpirun -np 1 ./fib-master.py $N 
}

loadMPI &&
run