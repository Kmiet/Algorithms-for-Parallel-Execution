#!/usr/bin/env python3
from mpi4py import MPI
import numpy
import sys

comm = MPI.COMM_SELF.Spawn(sys.executable, args=['./fib-worker.py'], maxprocs=32)

start_time = MPI.Wtime()

N = numpy.array(3, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)

FIB = numpy.array(0.0, 'i')
comm.Reduce(None, [FIB, MPI.INT], op=MPI.SUM, root=MPI.ROOT)

end_time = MPI.Wtime()
print(FIB)
print(end_time - start_time)

comm.Disconnect()