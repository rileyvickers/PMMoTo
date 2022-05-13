import numpy as np
from mpi4py import MPI
import pdb
import sys
import time
import PMMoTo
import math
import edt


def analyticDistance(x):
    return 0.01*math.cos(0.01*x) + 0.5*math.sin(x) + 0.75


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    start_time = time.time()

subDomains = [2,2,2]
nodes = [281,61,61]
#nodes = [560,120,120]
periodic = [False,False,False]
inlet  = [ 1,0,0]
outlet = [-1,0,0]

numSubDomains = np.prod(subDomains)

domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,periodic,"InkBotle",None,None)
sDEDTL = PMMoTo.calcEDT(rank,domain,sDL,sDL.grid)
drainL = PMMoTo.calcDrainage(rank,size,domain,sDL,inlet,sDEDTL)

if rank == 0:
    print("--- %s seconds ---" % (time.time() - start_time))
    sD = np.empty((numSubDomains), dtype = object)
    sDEDT = np.empty((numSubDomains), dtype = object)
    sDdrain = np.empty((numSubDomains), dtype = object)
    sD[0] = sDL
    sDEDT[0] = sDEDTL
    sDdrain[0] = drainL
    for neigh in range(1,numSubDomains):
        sD[neigh] = comm.recv(source=neigh)
        sDEDT[neigh] = comm.recv(source=neigh)
        sDdrain[neigh] = comm.recv(source=neigh)

if rank > 0:
    for neigh in range(1,numSubDomains):
        if rank == neigh:
            comm.send(sDL,dest=0)
            comm.send(sDEDTL,dest=0)
            comm.send(drainL,dest=0)


if rank ==0:
    pass

    #
    # ## Gen Single Domain
    # x = np.linspace(domain.domainSize[0,0] + domain.dX/2, domain.domainSize[0,1]-domain.dX/2, nodes[0])
    # y = np.linspace(domain.domainSize[1,0] + domain.dY/2, domain.domainSize[1,1]-domain.dY/2, nodes[1])
    # z = np.linspace(domain.domainSize[2,0] + domain.dZ/2, domain.domainSize[2,1]-domain.dZ/2, nodes[2])
    # max_z = 1.5
    # max_x = 14
    # scale_z = nodes[2]/(max_z * 2)
    # scale_x = max_x/nodes[0]
    # gridOut = np.zeros(nodes,dtype=np.int8)
    # for i in range(0,nodes[0]):
    #     for j in range(0,nodes[1]):
    #         for k in range(0,nodes[2]):
    #             x1=scale_x*i
    #             r = (0.01*math.cos(0.01*x1)+0.5*math.sin(x1)+0.75)*scale_z
    #             if (k-nodes[2]/2.)*(k-nodes[2]/2.) + (j-nodes[1]/2.)*(j-nodes[1]/2.) <= r*r:
    #                 gridOut[i,j,k]=1
    #
    # realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
    #
    #
    # ### Reconstruct SubDomains to Check EDT ####
    # checkGrid = np.zeros_like(realDT)
    # n = 0
    # for i in range(0,subDomains[0]):
    #     for j in range(0,subDomains[1]):
    #         for k in range(0,subDomains[2]):
    #             checkGrid[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
    #                       sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
    #                       sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
    #                       = sD[n].grid[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
    #                                       sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
    #                                       sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
    #             n = n + 1
    #
    # diffGrid = gridOut-checkGrid
    # print("L2 Grid Error Norm",np.linalg.norm(diffGrid) )
    #
    #
    # checkEDT = np.zeros_like(realDT)
    # n = 0
    # for i in range(0,subDomains[0]):
    #     for j in range(0,subDomains[1]):
    #         for k in range(0,subDomains[2]):
    #             checkEDT[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
    #                      sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
    #                      sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
    #                      = sDEDT[n].EDT[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
    #                                     sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
    #                                     sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
    #             n = n + 1
    #
    # diffEDT = np.abs(realDT-checkEDT)
    # print("L2 EDT Error Norm",np.linalg.norm(diffEDT) )
    #
    #
    # printGridOut = np.zeros([gridOut.size,7])
    # c = 0
    # for i in range(0,gridOut.shape[0]):
    #     for j in range(0,gridOut.shape[1]):
    #         for k in range(0,gridOut.shape[2]):
    #             printGridOut[c,0] = x[i]
    #             printGridOut[c,1] = y[j]
    #             printGridOut[c,2] = z[k]
    #             printGridOut[c,3] = gridOut[i,j,k]
    #             printGridOut[c,4] = realDT[i,j,k]
    #             c = c + 1
    #
    # header = "x,y,z,Grid,EDT,NWP,NWPFinal"
    # file = "dataDump/3dGridOut.csv"
    # np.savetxt(file,printGridOut, delimiter=',',header=header)
    #
    #
    # for nn in range(0,numSubDomains):
    #     printGridOut = np.zeros([sD[nn].grid.size,7])
    #     c = 0
    #     for i in range(0,sD[nn].grid.shape[0]):
    #         for j in range(0,sD[nn].grid.shape[1]):
    #             for k in range(0,sD[nn].grid.shape[2]):
    #                 printGridOut[c,0] = sD[nn].x[i]#sDAll[nn].indexStart[0] + i #sDAll[nn].x[i]
    #                 printGridOut[c,1] = sD[nn].y[j]#sDAll[nn].indexStart[1] + j#sDAll[nn].y[j]
    #                 printGridOut[c,2] = sD[nn].z[k]#sDAll[nn].indexStart[2] + k#sDAll[nn].z[k]
    #                 printGridOut[c,3] = sD[nn].grid[i,j,k]
    #                 printGridOut[c,4] = sDEDT[nn].EDT[i,j,k]
    #                 printGridOut[c,5] = sDdrain[nn].nwp[i,j,k]
    #                 printGridOut[c,6] = sDdrain[nn].nwpFinal[i,j,k]
    #                 c = c + 1
    #
    #     header = "x,y,z,Grid,EDT,NWP,NWPFinal"
    #     file = "dataDump/3dsubGridOut_"+str(nn)+".csv"
    #     np.savetxt(file,printGridOut, delimiter=',',header=header)
