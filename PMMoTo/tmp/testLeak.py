import os
import psutil
from mpi4py import MPI
import numpy as np
import resource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()



def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    mem2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return mem,mem2

def main_func():
    hist = np.arange(100000)
    print("Memory size of a NumPy array:",hist.nbytes)

    for _ in range(10):
        print(f"rank {rank}, memory usage = ",get_memory_usage() )
        for _ in range(1000):
            # case 0: allreduce
            # memory leak
            result = comm.allreduce(hist, op=MPI.SUM)
            # case 1: reduce
            # no memory leak
            # result = comm.reduce(hist, op=MPI.SUM, root=0)
            # case 2: Allreduce
            # no memory leak
            # result = np.empty_like(hist)
            # comm.Allreduce(hist, result, op=MPI.SUM)

        assert result[0] == 0
        assert result[1] == comm.size


if __name__ == "__main__":
    main_func()
