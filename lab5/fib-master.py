#!/usr/bin/env python3
from mpi4py import MPI
import numpy
import sys

argc = len(sys.argv)
if argc != 2:
  print("Missing argument - N. Usage: fib-master.py <n>\n")
  exit(1)

# Nth fib number
N = int(sys.argv[1])

comm = MPI.COMM_SELF.Spawn(sys.executable, args=['fib-worker.py'])

start_time = MPI.Wtime()

N = numpy.array(N, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)

FIB = numpy.array(0.0, 'i')
comm.Reduce(None, [FIB, MPI.INT], op=MPI.SUM, root=MPI.ROOT)

end_time = MPI.Wtime()
# print(FIB)
print(N, end_time - start_time)

comm.Disconnect()
exit(1)