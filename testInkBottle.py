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

subDomains = [4,1,1]
#nodes = [280,60,60]
#nodes = [560,120,120]
nodes = [840,180,180]
#nodes = [1120,240,240]
periodic = [False,False,False]
inlet  = [ 1,0,0]
outlet = [-1,0,0]

numSubDomains = np.prod(subDomains)

domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,periodic,"InkBotle",None,None)
if rank==0:
    print("Domain Gen --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid)
if rank==0:
    print("Distance --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
drainL = PMMoTo.calcDrainage(rank,size,domain,sDL,inlet,sDEDTL)
if rank==0:
    print("Drain --- %s seconds ---" % (time.time() - start_time))

# if rank == 0:
#     sD = np.empty((numSubDomains), dtype = object)
#     sDEDT = np.empty((numSubDomains), dtype = object)
#     sDdrain = np.empty((numSubDomains), dtype = object)
#     sD[0] = sDL
#     sDEDT[0] = sDEDTL
#     sDdrain[0] = drainL
#     for neigh in range(1,numSubDomains):
#         sD[neigh] = comm.recv(source=neigh)
#         sDEDT[neigh] = comm.recv(source=neigh)
#         sDdrain[neigh] = comm.recv(source=neigh)
#
# if rank > 0:
#     for neigh in range(1,numSubDomains):
#         if rank == neigh:
#             comm.send(sDL,dest=0)
#             comm.send(sDEDTL,dest=0)
#             comm.send(drainL,dest=0)
#
#
# if rank ==0:
#
#     ## Gen Single Domain
#     x = np.linspace(domain.domainSize[0,0] + domain.dX/2, domain.domainSize[0,1]-domain.dX/2, nodes[0])
#     y = np.linspace(domain.domainSize[1,0] + domain.dY/2, domain.domainSize[1,1]-domain.dY/2, nodes[1])
#     z = np.linspace(domain.domainSize[2,0] + domain.dZ/2, domain.domainSize[2,1]-domain.dZ/2, nodes[2])
#     gridOut = np.zeros(nodes,dtype=np.int8)
#     for i in range(0,nodes[0]):
#         for j in range(0,nodes[1]):
#             for k in range(0,nodes[2]):
#                 r = (0.01*math.cos(0.01*x[i])+0.5*math.sin(x[i])+0.75)
#                 if (z[k]*z[k] + y[j]*y[j]) <= r*r:
#                     gridOut[i,j,k]=1
#
#
#     print(domain.dX, domain.dY, domain.dZ)
#     realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
#
#     ### Reconstruct SubDomains to Check EDT ####
#     checkGrid = np.zeros_like(realDT)
#     n = 0
#     for i in range(0,subDomains[0]):
#         for j in range(0,subDomains[1]):
#             for k in range(0,subDomains[2]):
#                 checkGrid[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
#                           sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
#                           sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
#                           = sD[n].grid[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
#                                           sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
#                                           sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
#                 n = n + 1
#
#     diffGrid = gridOut-checkGrid
#     print("L2 Grid Error Norm",np.linalg.norm(diffGrid) )
#
#
#     checkEDT = np.zeros_like(realDT)
#     n = 0
#     for i in range(0,subDomains[0]):
#         for j in range(0,subDomains[1]):
#             for k in range(0,subDomains[2]):
#                 checkEDT[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
#                          sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
#                          sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
#                          = sDEDT[n].EDT[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
#                                         sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
#                                         sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
#                 n = n + 1
#
#     diffEDT = np.abs(realDT-checkEDT)
#     print("L2 EDT Error Norm",np.linalg.norm(diffEDT) )

#
#     printGridOut = np.zeros([gridOut.size,7])
#     c = 0
#     for i in range(0,gridOut.shape[0]):
#         for j in range(0,gridOut.shape[1]):
#             for k in range(0,gridOut.shape[2]):
#                 printGridOut[c,0] = x[i]
#                 printGridOut[c,1] = y[j]
#                 printGridOut[c,2] = z[k]
#                 printGridOut[c,3] = gridOut[i,j,k]
#                 printGridOut[c,4] = realDT[i,j,k]
#                 printGridOut[c,5] = diffEDT[i,j,k]
#                 c = c + 1
#
#     header = "x,y,z,Grid,EDT,diffEDT,No"
#     file = "dataDump/3dGridOut.csv"
#     np.savetxt(file,printGridOut, delimiter=',',header=header)
#
#
#     for nn in range(0,numSubDomains):
#         printGridOut = np.zeros([sD[nn].grid.size,7])
#         c = 0
#         for i in range(0,sD[nn].grid.shape[0]):
#             for j in range(0,sD[nn].grid.shape[1]):
#                 for k in range(0,sD[nn].grid.shape[2]):
#                     printGridOut[c,0] = sD[nn].x[i]#sDAll[nn].indexStart[0] + i #sDAll[nn].x[i]
#                     printGridOut[c,1] = sD[nn].y[j]#sDAll[nn].indexStart[1] + j#sDAll[nn].y[j]
#                     printGridOut[c,2] = sD[nn].z[k]#sDAll[nn].indexStart[2] + k#sDAll[nn].z[k]
#                     printGridOut[c,3] = sD[nn].grid[i,j,k]
#                     printGridOut[c,4] = sDEDT[nn].EDT[i,j,k]
#                     printGridOut[c,5] = sDdrain[nn].nwp[i,j,k]
#                     printGridOut[c,6] = sDdrain[nn].nwpFinal[i,j,k]
#                     c = c + 1
#
#         header = "x,y,z,Grid,EDT,NWP,NWPFinal"
#         file = "dataDump/3dsubGridOut_"+str(nn)+".csv"
#         np.savetxt(file,printGridOut, delimiter=',',header=header)
#
#     # nn = 0
#     # c = 0
#     # data = np.empty((0,3))
#     # orient = sD[0].Orientation
#     # for fIndex in orient.faces:
#     #     #orientID = orient.faces[fIndex]['ID']
#     #     #for procs in sDEDT[nn].solidsAll.keys():
#     #         #for fID in sDEDT[nn].solidsAll[procs]['orientID'].keys():
#     #             #if fID == orientID:
#     #     data = np.append(data,sDEDT[nn].trimmedSolids[fIndex],axis=0)
#     #                 #data = np.append(data,sDEDT[nn].solidsAll[procs]['orientID'][fID],axis=0)
#     # printGridOut = np.zeros([data.shape[0],3])
#     # dX = 0.05
#     # dY = 0.05
#     # dZ = 0.05
#     # print(data.shape[0])
#     # for i in range(data.shape[0]):
#     #     printGridOut[c,0] = data[i,0]*dX
#     #     printGridOut[c,1] = data[i,1]*dY - 1.5
#     #     printGridOut[c,2] = data[i,2]*dZ - 1.5
#     #     # printGridOut[c,0] = data[i,0]
#     #     # printGridOut[c,1] = data[i,1]
#     #     # printGridOut[c,2] = data[i,2]
#     #     c = c + 1
#     #
#     # header = "x,y,z"
#     # file = "dataDump/3dsubGridSINGLEAllSolids_"+str(nn)+".csv"
#     # np.savetxt(file,printGridOut, delimiter=',',header=header)
