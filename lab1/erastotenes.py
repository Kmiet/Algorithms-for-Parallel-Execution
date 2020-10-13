#!/usr/bin/env python
from mpi4py import MPI
import sys
import os
import math


def is_divisible_by(x):
  return lambda y : True if y % x == 0 else False


def erastotenes_loop(arr, dividers):
  if not arr or not dividers or not arr[-1] > dividers[0]:
    return arr
  arr = list(filter(lambda x : x == dividers[0] or not is_divisible_by(dividers[0])(x), arr))
  return erastotenes_loop(arr, dividers[1:])


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
  data = range(2, N+1)

  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  data = comm.scatter(data, root=0)
  data = erastotenes(data)
  print(rank, data)
  data = comm.gather(data, root=0)
  if rank == 0:
    print(data)