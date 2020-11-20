#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import os
import math

def read_star_file():
  global N
  star_count = 0;
  stars = []

  with open('stars.txt') as f:
    line = f.readline()
    while line:
      M, Rx, Ry, Rz, Vx, Vy, Vz = line.split(" ")
      stars.append([M, Rx, Ry, Rz, Vx, Vy, Vz])
      star_count += 1
      line = f.readline()

  if N != star_count:
    N = star_count

  stars = np.array(stars, dtype=np.double).reshape((N, 7))

  return stars


def save_to_file(stars):
  with open('stars.txt', 'w+') as f:
    for s in stars:
      M, Rx, Ry, Rz, Vx, Vy, Vz = s
      print(M, Rx, Ry, Rz, Vx, Vy, Vz, file=f)


def gen_star_data():
  global N
  stars = np.random.rand(N, 7)

  return stars


def calculate_forces(stars, neighbour_stars, acc=None, same_stars=False):
  global G

  F = np.zeros((len(stars), 3))

  for i, s1 in enumerate(stars):
    Mi, Rxi, Ryi, Rzi, Vxi, Vyi, Vzi = s1
    Ri = np.array([Rxi, Ryi, Rzi])

    for j, s2 in enumerate(neighbour_stars):
      if not same_stars or i != j:
        Mj, Rxj, Ryj, Rzj, Vxj, Vyj, Vzj = s2
        Rj = np.array([Rxj, Ryj, Rzj])

        F[i] += Mj / (math.sqrt((Rxj - Rxi) ** 2 + (Ryj - Ryi) ** 2 + (Rzj - Rzi) ** 2)) * (Ri - Rj)

    F[i] *= G * Mi
  
  if acc:
    F += acc

  return F


if __name__ == "__main__":
  global N, G
  argc = len(sys.argv)
  if argc != 2:
    print("Missing argument - N. Usage: n-body-problem.py <n>\n")
    exit(1)

  # Star count
  N = int(sys.argv[1])
  G = 6.67

  comm = MPI.COMM_WORLD
  world_size = comm.Get_size()
  rank = comm.Get_rank()

  left_neighbour = rank - 1 if rank != 0 else world_size - 1
  right_neighbour = rank + 1 if rank != (world_size - 1) else 0
  
  stars_per_node = int(N / world_size)

  # buffer_sizes = [stars_per_node if i + 1 != world_size else N - (i * stars_per_node) for i in range(world_size)]

  stars = None
  if rank == 0:
    stars = read_star_file()
    # stars = gen_star_data()
    # save_to_file(stars)
    stars = np.array_split(stars, world_size)
  stars = comm.scatter(stars, root=0)

  # recvbuff = np.zeros((N, 3))

  # F = np.zeros((N, 3))

  start_time = MPI.Wtime()
  F = calculate_forces(stars, stars, same_stars=True)

  star_buffer = np.zeros(stars_per_node + 1)
  star_buffer[:stars_per_node] = stars
  star_buffer[-1] = rank

  for _ in range(world_size - 1):
    comm.Send([star_buffer, MPI.DOUBLE], dest=right_neighbour)
    comm.Recv([star_buffer, MPI.DOUBLE], source=left_neighbour)

    new_stars = star_buffer[:stars_per_node] 
    star_owner = star_buffer[-1]

    F = calculate_forces(stars, new_stars, acc=F)

  F = comm.gather(F, root=0)
  # comm.Gatherv(F, (recvbuff, buffer_sizes, None, MPI.DOUBLE), root=0)
  if rank == 0:
    print(F)

  # # print(rank, data)
  # end_time = MPI.Wtime()
  # comm.Barrier()

  # if rank == 0:
  #   print(N * M, world_size, end_time - start_time)