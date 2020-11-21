#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import os
import math

def read_star_file(fname=None):
  global N
  filenam = fname if fname is not None else 'stars.txt'
  star_count = 0;
  stars = []

  with open(filenam) as f:
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
  
  if acc is not None:
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
  
  star_size = 7
  force_size = 3
  stars_per_node = int(N / world_size)

  fname = 'stars_' + str(N) + '.txt'

  stars = None
  if rank == 0:
    stars = read_star_file(fname)
    # stars = gen_star_data()
    # save_to_file(stars)
    stars = np.array_split(stars, world_size)
  stars = comm.scatter(stars, root=0)
  start_time = MPI.Wtime()
  
  star_buffer_size = stars_per_node * star_size
  acc_size = stars_per_node * force_size
  F = np.zeros((stars_per_node, force_size))
  
  star_buffer = np.zeros(star_buffer_size + acc_size + 1)
  star_buffer[:star_buffer_size] = np.reshape(stars, stars_per_node * star_size)
  star_buffer[star_buffer_size:star_buffer_size + acc_size] = np.reshape(F, acc_size)
  star_buffer[-1] = rank

  # stars = np.reshape(stars, (stars_per_node, star_size))

  print(rank, 'star_buff', left_neighbour, right_neighbour)

  for _ in range(int(world_size / 2)):
    comm.Send([star_buffer, MPI.DOUBLE], dest=left_neighbour)
    comm.Recv([star_buffer, MPI.DOUBLE], source=right_neighbour)

    print(rank, 'comm')

    new_stars = star_buffer[:star_buffer_size] 
    Facc = star_buffer[star_buffer_size:star_buffer_size + acc_size]

    print(rank, 'facc')

    tmp_F = calculate_forces(stars, np.reshape(new_stars, (stars_per_node, star_size)))
    print(rank, 'tmpF')
    F += tmp_F
    star_buffer[star_buffer_size:star_buffer_size + acc_size] = Facc + np.reshape(tmp_F, acc_size)
    print(rank, True)

  comm.Send([star_buffer, MPI.DOUBLE], dest=star_buffer[-1])
  comm.Recv([star_buffer, MPI.DOUBLE], source=(star_buffer[-1] + 1) % world_size)

  Facc = star_buffer[star_buffer_size:star_buffer_size + acc_size]
  tmp_F = calculate_forces(stars, stars, same_stars=True)
  F += tmp_F
  F -= np.reshape(Facc, (stars_per_node, force_size))

  # print(rank, F)

  F = comm.gather(F, root=0)
  end_time = MPI.Wtime()
  comm.Barrier()

  if rank == 0:
    # print(np.array(F).tolist())
    print(N, world_size, end_time - start_time)