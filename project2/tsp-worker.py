#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys

INF = 1000000
STOP_SPLIT_LEVEL = 2
MAX_BUFFER_SIZE = 3 + 100 * 100 + 100


class Node:
  def __init__(self, city_id=None, parent=None, children=[]):
    self.visited = False

    self.city_id = city_id
    self.lower_bound = 0 # min distance, might be unachievable
    self.upper_bound = INF # current best distance

    self.parent = parent
    self.children = {}
    self.best_child = None

    for c in children:
      cities_left = set(children)
      cities_left.remove(c)
      self.children[c] = Node(city_id=c, parent=self, children=list(cities_left))

  def is_leaf(self):
    return not self.children

  def is_root(self):
    return not self.parent

  def get_child(self, id):
    return self.children[id]

  def get_parent(self):
    return self.parent

  def set_best_child(self, id):
    self.best_child = id

  def set_bounds(self, lower=None, upper=None):
    if lower:
      self.lower_bound = lower
    if upper:
      self.upper_bound = upper

  def set_visited(self, visited):
    self.visited = visited


def branch_n_bound_tsp(distances, city_count, city_id, cities_left):
  distances = distances.reshape((city_count, city_count))

  root_node = Node(city_id=city_id, children=cities_left)
  current_node = root_node
  parent_node = current_node

  stack = [current_node]
  
  # DFS-branch-n-bound

  while stack:
    current_node = stack.pop()
    if not current_node.is_root():
      parent_node = current_node.get_parent()

    absolute_min_dist = distances[current_node.city_id, parent_node.city_id] + parent_node.lower_bound
    # print(parent_node.city_id, current_node.city_id, current_node.lower_bound, current_node.upper_bound, absolute_min_dist)

    if current_node.visited:
      if not current_node.is_root() and current_node.upper_bound < parent_node.upper_bound:
        parent_node.set_bounds(upper=current_node.upper_bound)
        parent_node.set_best_child(current_node.city_id)

    # check if worth visiting
    elif absolute_min_dist < parent_node.upper_bound:
      current_node.set_visited(True)
      if absolute_min_dist > 0:
        current_node.set_bounds(lower=absolute_min_dist)
      
      if current_node.is_leaf():
        absolute_min_dist += distances[current_node.city_id, 0]
        current_node.set_bounds(upper=absolute_min_dist)

        if absolute_min_dist < parent_node.upper_bound:
          parent_node.set_bounds(upper=absolute_min_dist)
          parent_node.set_best_child(current_node.city_id)
      
      else:
        stack.append(current_node)
        for _, child in reversed(current_node.children.items()):
          stack.append(child)

  best_path = []
  current_node = root_node
  while current_node:
    best_path.append(current_node.city_id)
    if not current_node.is_leaf():
      current_node = current_node.get_child(current_node.best_child)
    else:
      current_node = None
      best_path.append(0)

  return root_node.upper_bound, np.array(best_path)


if __name__ == "__main__":
  comm = MPI.Comm.Get_parent()
  size = comm.Get_size()
  rank = comm.Get_rank()
  rank_world = MPI.COMM_WORLD.Get_rank()

  buffer = np.zeros(MAX_BUFFER_SIZE, dtype='i')
  comm.Bcast([buffer, MPI.INT], root=0)

  buffer_size = len(buffer)

  level = buffer[0]
  city_id = buffer[1]
  city_count = buffer[2]

  distances_end_index = 3 + city_count ** 2

  cities_left = np.array( list( filter(lambda x : x > 0, buffer[distances_end_index:]) ) )
  cities_left_count = len(cities_left)

  if not (level < STOP_SPLIT_LEVEL):
    best_cost, best_path = branch_n_bound_tsp(
      buffer[3 : distances_end_index],
      city_count,
      city_id,
      cities_left
    )
    
    results = np.zeros(city_count + 2, dtype='i')
    results[-1] = best_cost
    results[-1 * (cities_left_count + 3) : -1] = best_path
    # print(results, best_path, best_cost)
    
    comm.Gather([results, MPI.INT], None, root=0)

  else:
    sub_comms = {}
    buffer[0] = level + 1
    buffer[distances_end_index + cities_left_count - 1] = -1

    for c in cities_left:
      tmp_cities = set(cities_left)
      tmp_cities.remove(c)

      buffer[1] = c
      buffer[distances_end_index : (distances_end_index + cities_left_count - 1)] = list(tmp_cities)

      sub_comm = MPI.COMM_SELF.Spawn(sys.executable, args=['tsp-worker.py'])
      sub_comms[c] = sub_comm

      sub_comm.Bcast([buffer, MPI.INT], root=MPI.ROOT)

    results = np.zeros(city_count + 2, dtype='i')
    best_cost = INF

    for child_city_id, sc in sub_comms.items():
      X = np.zeros(city_count + 2, 'i')
      sc.Gather(None, [X, MPI.INT], root=MPI.ROOT)

      curr_cost = X[-1] + buffer[3 : distances_end_index][city_id * city_count + child_city_id]

      if curr_cost < best_cost:
        results[:-1] = X[:-1]
        results[level - 1] = city_id
        best_cost = curr_cost

    results[-1] = best_cost
    comm.Gather([results, MPI.INT], None, root=0)

  comm.Disconnect()
  exit(1)