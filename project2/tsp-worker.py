#!/usr/bin/env python3
from mpi4py import MPI
from queue import Queue
import numpy as np
import sys

INF = 1000000
STOP_SPLIT_LEVEL = 2

# xd

if __main__ == "__main__":
  comm = MPI.Comm.Get_parent()
  size = comm.Get_size()
  rank = comm.Get_rank()
  rank_world = MPI.COMM_WORLD.Get_rank()

  buffer = np.array(dtype='i')
  comm.Bcast([buffer, MPI.INT], root=0)

  buffer_size = len(buffer)

  level = buffer[0]
  city_id = buffer[1]
  city_count = buffer[2]

  distances_end_index = 3 + city_count ** 2

  cities_left = np.array( list( filter(buffer[distances_end_index : -1], lambda x : x > 0) ) )
  cities_left_count = len(cities_left)

  print("WORKER", level, city_id, cities_left)

  if not (level < STOP_SPLIT_LEVEL):
    # distances = buffer[3 : distances_end_index]
    ONE = np.array([1, 1, city_id, 0, 13], dtype='i')
    comm.Gather([ONE, MPI.INT], None, root=0)
    # branch_n_bound_tsp(distances)

  else:
    sub_comms = []
    buffer[distances_end_index + cities_left_count - 1] = -1

    for c in cities_left:
      tmp_cities = set(cities_left)
      tmp_cities.remove(c)

      buffer[1] = c
      buffer[distances_end_index : (distances_end_index + cities_left_count - 1)] = list(tmp_cities)

      sub_comm = MPI.COMM_SELF.Spawn(sys.executable, args=['tsp-worker.py'])
      sub_comms.append(sub_comm)

      sub_comm.Bcast([buffer, MPI.INT], root=MPI.ROOT)

    results = np.zeros(city_count + 1, dtype='i')
    best_cost = INF

    for sc in sub_comms:
      X = np.zeros(city_count + 1, 'i')
      sc.Gather(None, [X, MPI.INT], root=MPI.ROOT)

      if X[-1] < best_cost:
        results[0 : city_count] = X[0 : city_count]
        best_cost = X[-1]

    results[-1] = best_cost
    comm.Gather([results, MPI.INT], None, root=0)

  comm.Disconnect()
  exit(1)