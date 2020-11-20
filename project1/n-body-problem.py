#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import os
import math


def read_star_file():
  global N
  star_count = 0;
  M, Rx, Ry, Rz, Vx, Vy, Vz = [[] for _ in range(7)]

  with open('stars.txt') as f:
    line = f.readline()
    while line:
      star_count += 1
      f.readline()

  if N != star_count:
    N = star_count

  return M, Rx, Ry, Rz, Vx, Vy, Vz


def gen_star_data():
  global N
  return []


def gauss_seidel(upper):
  return upper


if __name__ == "__main__":
  global N
  argc = len(sys.argv)
  if argc != 2:
    print("Missing argument - N. Usage: gauss_seidel.py <n> <m> <efficiency> <conductivity> <epsilon> \n")
    exit(1)

  # Star count
  N = int(sys.argv[1])

  comm = MPI.COMM_WORLD
  world_size = comm.Get_size()
  rank = comm.Get_rank()

  left_neighbour = rank - 1 if rank != 0 else world_size - 1
  right_neighbour = rank + 1 if rank != (world_size - 1) else 0
  
  stars_per_node = int(N / world_size)
  
  buffer_sizes = [stars_per_node if i + 1 != world_size else N - (i * stars_per_node) for i in range(world_size)]
  
  M, Rx, Ry, Rz, Vx, Vy, Vz = [[] for _ in range(7)] 

  if rank == 0:
    M, Rx, Ry, Rz, Vx, Vy, Vz = gen_star_data()

  recvbuff = np.full(problem_size, b)

  offset = rank * stars_per_node
  M = M[offset : (offset + buffer_sizes[rank])]
  Rx = Rx[offset : (offset + buffer_sizes[rank])]
  Ry = Ry[offset : (offset + buffer_sizes[rank])]
  Rz = Rz[offset : (offset + buffer_sizes[rank])]
  Vx = Vx[offset : (offset + buffer_sizes[rank])]
  Vy = Vy[offset : (offset + buffer_sizes[rank])]
  Vz = Vz[offset : (offset + buffer_sizes[rank])]

  start_time = MPI.Wtime()

  

  while True:
    # communication
    if not upper_last_iteration:
      row_buffer[:N] = data[:N]
      row_buffer[-1] = int(is_last_iteration)
      comm.Send([row_buffer, MPI.DOUBLE], dest=rank-1)

    if not lower_last_iteration:
      row_buffer_2[:N] = data[-N:]
      row_buffer_2[-1] = int(is_last_iteration)
      comm.Send([row_buffer_2, MPI.DOUBLE], dest=rank+1)

    if not upper_last_iteration:
      comm.Recv([row_buffer, MPI.DOUBLE], source=rank-1)
      upper[:N] = row_buffer[:N]
      upper_last_iteration = int(row_buffer[-1]) != 0

    if not lower_last_iteration:
      comm.Recv([row_buffer_2, MPI.DOUBLE], source=rank+1)
      lower[:N] = row_buffer_2[:N]
      lower_last_iteration = int(row_buffer_2[-1]) != 0

    if is_last_iteration:
      break

    is_last_iteration = gauss_seidel(upper, lower, phase=phase, epsilon=e)

  comm.Gatherv(data, (recvbuff, buffer_sizes, None, MPI.DOUBLE), root=0)
  if rank == 0:

  # print(rank, data)
  end_time = MPI.Wtime()
  comm.Barrier()

  if rank == 0:
    print(N * M, world_size, end_time - start_time)
    # print("RESULT")
    # print(recvbuff.reshape((M, N)))