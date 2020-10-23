#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import os
import math

RED = 0
BLACK = 1

def gauss_seidel(upper, lower, phase=0, epsilon=0.001):
  global data, b, N, M, problem_size
  is_last_iteration = True

  phase_size = int(problem_size / 2)
  if phase == RED:
    phase_size += problem_size % 2

  # red phase
  for j in range(phase_size):
    i = j * 2 + phase
    up = upper[j % N]
    left = 0
    right = 0
    down = lower[j % N]

    if i % N != 0:
      left = data[i-1]
    if (i + 1) % N != 0:
      right = data[i+1]
    if i / N != 0:
      up = data[i - N]
    if i < (problem_size - N):
      down = data[i + N]
    
    new_val = (b - left - up - right - down) / -4
    is_last_iteration = is_last_iteration and (new_val - data[i] < epsilon)
    data[i] = new_val

  return is_last_iteration


if __name__ == "__main__":
  global data, b, N, M, problem_size
  argc = len(sys.argv)
  if argc < 6:
    print("Missing argument - N. Usage: gauss_seidel.py <n> <m> <efficiency> <conductivity> <epsilon> \n")
    exit(1)

  # X - axis
  N = int(sys.argv[1])
  # Y - axis
  M = int(sys.argv[2])
  # efficiency
  G = float(sys.argv[3])
  # conductivity
  L = float(sys.argv[4])
  # epsilon
  e = float(sys.argv[5])

  comm = MPI.COMM_WORLD
  world_size = comm.Get_size()
  rank = comm.Get_rank()
  
  b = - 1.0 * G / L
  problem_size = M * N
  rows_per_node = int(M / world_size)
  
  buffer_sizes = [rows_per_node * N if i + 1 != world_size else (M - rank * rows_per_node) * N for i in range(world_size)]
  recvbuff = np.full(problem_size, b)
  problem_size = buffer_sizes[rank]

  offset = rank * rows_per_node * N
  data = recvbuff[offset : (offset + buffer_sizes[rank])]

  upper = np.zeros(N)
  lower = np.zeros(N)

  row_buffer = np.empty(N + 1)

  is_last_iteration = False
  upper_last_iteration = False
  lower_last_iteration = False

  if rank == 0:
    upper_last_iteration = True
  if rank == (world_size - 1):
    lower_last_iteration = True

  phase = RED

  while True:
    # communication
    if not upper_last_iteration:
      row_buffer[:N] = data[:N]
      row_buffer[-1] = int(is_last_iteration)
      comm.Send([row_buffer, MPI.DOUBLE], dest=rank-1)

    if not lower_last_iteration:
      row_buffer[:N] = data[-N:]
      row_buffer[-1] = int(is_last_iteration)
      comm.Send([row_buffer, MPI.DOUBLE], dest=rank+1)

    if not upper_last_iteration:
      comm.Recv([row_buffer, MPI.DOUBLE], source=rank-1)
      upper[:N] = row_buffer[:N]
      upper_last_iteration = int(row_buffer[-1]) != 0

    if not lower_last_iteration:
      comm.Recv([row_buffer, MPI.DOUBLE], source=rank+1)
      lower[:N] = row_buffer[:N]
      lower_last_iteration = int(row_buffer[-1]) != 0

    if is_last_iteration:
      break

    is_last_iteration = gauss_seidel(upper, lower, phase=phase, epsilon=e)

    if rank > int(world_size / 2):
      is_last_iteration = is_last_iteration and lower_last_iteration
    elif rank < int(world_size / 2):
      is_last_iteration = is_last_iteration and upper_last_iteration
    else:
      is_last_iteration = is_last_iteration and upper_last_iteration and lower_last_iteration
    
    if phase == RED:
      phase = BLACK
    else:
      phase = RED

  comm.Gatherv(data, (recvbuff, buffer_sizes, None, MPI.DOUBLE), root=0)
  # print(rank, data)
  # comm.Barrier()

  if rank == 0:
    print("RESULT")
    print(recvbuff.reshape((M, N)))