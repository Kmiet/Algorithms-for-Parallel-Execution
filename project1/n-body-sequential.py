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


def save_to_file(stars, fname=None):
  filenam = fname if fname is not None else 'stars.txt'
  with open(filenam, 'w+') as f:
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
  
  star_size = 7

  fname='stars_' + str(N) + '.txt'

  stars = read_star_file(fname)
  # if rank == 0:
  #   save_to_file(stars, fname)

  start_time = MPI.Wtime()
  F = calculate_forces(stars, stars, same_stars=True)
  end_time = MPI.Wtime()

  if rank == 0:
    print(N, world_size, end_time - start_time)