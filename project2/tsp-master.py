#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys


def count_cities(filename):
  N = 0
  with open(filename, 'r+') as f:
    line = f.readline()
    while line:
      N += 1
      line = f.readline()
  return N


def read_distances(filename):
  distance_matrix = []
  with open(filename, 'r+') as f:
    line = f.readline()
    while line:
      distance_matrix.append([int(d) for d in line.split(",")])
      line = f.readline()

  return np.array(distance_matrix).flatten()


if __name__ == "__main__":
  argc = len(sys.argv)
  if argc != 2:
    print("Missing argument - filename. Usage: tsp-master.py <filename>\n")
    exit(1)

  # Nth fib number
  city_count = count_cities(sys.argv[1])
  distances_end_index = 3 + city_count ** 2

  buffer = np.zeros(distances_end_index + city_count, dtype='i')

  buffer[0] = 1
  buffer[1] = 0
  buffer[2] = city_count
  buffer[3 : distances_end_index] = read_distances(sys.argv[1])
  buffer[distances_end_index : -1] = np.array(range(1, city_count))

  comm = MPI.COMM_SELF.Spawn(sys.executable, args=['tsp-worker.py'])

  start_time = MPI.Wtime()
  
  comm.Bcast([buffer, MPI.INT], root=MPI.ROOT)

  results = np.zeros(city_count + 2, 'i')
  comm.Gather(None, [results, MPI.INT], root=MPI.ROOT)

  end_time = MPI.Wtime()
  print("MASTER - BEST COST", results[-1])
  print("MASTER - BEST PATH", results[:-1])
  print("MASTER - TIME", end_time - start_time)

  comm.Disconnect()
  exit(1)