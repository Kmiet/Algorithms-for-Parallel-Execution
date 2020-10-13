#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import os
import math


def is_divisible_by(x):
  return lambda y : True if y % x == 0 else False


def erastotenes_loop(arr, dividers):
  res = arr
  for d in dividers:
    if not res[-1] > d:
      return res
    res = list(filter(lambda x : x == d or not is_divisible_by(d)(x), res))
  return res


def erastotenes(arr):
  if not arr:
    return arr
  
  dividers = range(2, int(math.floor(math.sqrt(arr[-1]))) + 1)
  dividers = erastotenes_loop(dividers, dividers)

  return erastotenes_loop(arr, dividers)


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Missing argument - N. Usage: erastotenes.py <num> \n")
    exit(1)

  N = int(sys.argv[1])

  comm = MPI.COMM_WORLD
  world_size = comm.Get_size()
  rank = comm.Get_rank()

  size_per_node = int(N / world_size)

  buffer_sizes = [size_per_node if i + 1 != world_size else N - i * size_per_node - 1 for i in range(world_size)]
  recvbuff = np.zeros(N - 1, dtype='i')

  offset = 2 + rank * size_per_node
  data = range(offset, offset + buffer_sizes[rank])

  data = erastotenes(data)

  comm.Gatherv(np.array(data, dtype='i'), (recvbuff, buffer_sizes, None, MPI.INT), root=0)
  print(rank, data)

  if rank == 0:
    non_zero = lambda x : x != 0
    data = list(filter(non_zero, recvbuff))
    print(data)