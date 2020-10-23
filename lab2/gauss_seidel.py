#!/usr/bin/env python3
# from mpi4py import MPI
import numpy as np
import sys
import os
import math

RED = 0
BLACK = 1

def is_divisible_by(x):
  return lambda y : True if y % x == 0 else False


def erastotenes_loop(arr, dividers):
  res = arr
  for d in dividers:
    if not res[-1] > d:
      return res
    res = list(filter(lambda x : x == d or not is_divisible_by(d)(x), res))
  return res


def gauss_seidel(arr, b, upper, lower, size, epsilon=0.001):
  data = arr
  n, m = size
  ps = n * m

  is_last_iteration = True
  row_id = 0


  # red phase
  for j in range(ps / 2 + ps % 2):
    i = j * 2 + RED
    up = 0
    left = 0
    right = 0
    down = 0

    if i % N != 0:
      left = data[i-1]
    if (i + 1) % N != 0:
      right = data[i+1]
    if i / n != 0:
      up = data[i - n]
    if i > (ps - n):
      down = data[i + n]
    
    new_val = (b - left - up - right - down) / -4
    if new_val - data[i] < epsilon:
      is_last_iteration = is_last_iteration and True
    data[i] = 

  dividers = range(2, int(math.floor(math.sqrt(arr[-1]))) + 1)
  dividers = erastotenes_loop(dividers, dividers)

  return data[1:-1]


if __name__ == "__main__":
  N = 4
  M = 5
  G = 0.8
  L = 2.3
  e = 0.001
  b = - 1.0 * G / L

  problem_size = M * N
  recvbuff = np.full(problem_size, b)

  data = recvbuff

  M_zeros = np.zeros(N)

  is_last_iteration = False

  while is_last_iteration:
    data, is_last_iteration = gauss_seidel(data, b, upper, lower, (N, M), epsilon=e)

  print(data)


# if __name__ == "__main__":
#   if len(sys.argv) < 5:
#     print("Missing argument - N. Usage: erastotenes.py <n> <m> <efficiency> <conductivity> \n")
#     exit(1)

#   # X - axis
#   N = int(sys.argv[1])
#   # Y - axis
#   M = int(sys.argv[2])
#   # efficiency
#   G = float(sys.argv[3])
#   # conductivity
#   L = float(sys.argv[4])

#   comm = MPI.COMM_WORLD
#   world_size = comm.Get_size()
#   rank = comm.Get_rank()

#   # number of points in grid
#   problem_size = M * N

#   rows_per_node = int(M / world_size)

#   # number of rows for each 'stripe'
#   buffer_sizes = [rows_per_node if i + 1 != world_size else M - rank * rows_per_node for i in range(world_size)]

#   recvbuff = np.full(problem_size, G / L)

#   offset = rank * size_per_node
#   data = recvbuff[offset : (offset + buffer_sizes[rank] -1)]

#   data = gauss_seidel(data)

#   comm.Gatherv(data, (recvbuff, buffer_sizes, None, MPI.DOUBLE), root=0)
#   print(rank, data)

#   if rank == 0:
#     non_zero = lambda x : x != 0
#     data = list(filter(non_zero, recvbuff))
#     print(data)