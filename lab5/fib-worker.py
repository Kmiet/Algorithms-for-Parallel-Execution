#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

N = numpy.array(0, dtype='i')

comm.Bcast([N, MPI.INT], root=0)
   
if (N < 2) 
  comm.Reduce([N, MPI.INT], None, op=MPI.SUM, root=0)
else {
  sub_comm1 = MPI.COMM_SELF.Spawn(sys.executable, args=['fib-worker.py'], maxprocs=32)
  sub_comm2 = MPI.COMM_SELF.Spawn(sys.executable, args=['fib-worker.py'], maxprocs=32)

  sub_comm1.Bcast([N - 1, MPI.INT], root=MPI.ROOT)
  sub_comm2.Bcast([N - 2, MPI.INT], root=MPI.ROOT)

  X = numpy.array(0.0, 'i')
  X = sub_comm1.Reduce(None, [X, MPI.INT], op=MPI.SUM, root=MPI.ROOT)

  Y = numpy.array(0.0, 'i')
  Y = sub_comm2.Reduce(None, [Y, MPI.INT], op=MPI.SUM, root=MPI.ROOT)

  comm.Reduce([X + Y, MPI.INT], None, op=MPI.SUM, root=0)
}

comm.Disconnect()