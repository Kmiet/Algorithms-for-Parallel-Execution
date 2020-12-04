#!/usr/bin/env python3
from mpi4py import MPI
import numpy
import sys

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
rank_world = MPI.COMM_WORLD.Get_rank()

N = numpy.array(0, dtype='i')

comm.Bcast([N, MPI.INT], root=0)

if (N < 3):
  ONE = numpy.array(1, dtype='i')
  comm.Reduce([ONE, MPI.INT], None, op=MPI.SUM, root=0)
else:
  sub_comm1 = MPI.COMM_SELF.Spawn(sys.executable, args=['fib-worker.py'])
  N_LESS_ONE = numpy.array(N - 1, dtype='i')
  sub_comm1.Bcast([N_LESS_ONE, MPI.INT], root=MPI.ROOT)

  sub_comm2 = MPI.COMM_SELF.Spawn(sys.executable, args=['fib-worker.py'])
  N_LESS_TWO = numpy.array(N - 2, dtype='i')
  sub_comm2.Bcast([N_LESS_TWO, MPI.INT], root=MPI.ROOT)

  X = numpy.array(0.0, 'i')
  sub_comm1.Reduce(None, [X, MPI.INT], op=MPI.SUM, root=MPI.ROOT)

  Y = numpy.array(0.0, 'i')
  sub_comm2.Reduce(None, [Y, MPI.INT], op=MPI.SUM, root=MPI.ROOT)

  FIB_N = numpy.array(X + Y, 'i')
  comm.Reduce([FIB_N, MPI.INT], None, op=MPI.SUM, root=0)

comm.Disconnect()
exit(1)