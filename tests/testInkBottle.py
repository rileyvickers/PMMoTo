import numpy as np
from mpi4py import MPI
import pdb
import sys
import time
import PMMoTo
import math
import edt
from scipy.ndimage import distance_transform_edt
from line_profiler import LineProfiler


import cProfile

def profile(filename=None, comm=MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator

@profile(filename="profile_out")
def my_function():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:
        start_time = time.time()

    subDomains = [2,2,2]
    nodes = [280,60,60]
    #nodes = [560,120,120]
    #nodes = [840,181,181]
    #nodes = [1120,240,240]
    #nodes = [2240,481,481]
    periodic = [False,False,False]
    inlet  = [ 1,0,0]
    outlet = [-1,0,0]

    pC = [1.65002]
    # pC = [7.69409,7.69409,7.69393,7.60403,7.36005,6.9909,6.53538,6.03352,5.52008,5.02123,
    #       4.55421,4.12854,3.74806,3.41274,3.12024,2.86704,2.64914,2.4625,2.30332,2.16814,2.05388,
    #       1.95783,1.87764,1.81122,1.75678,1.7127,1.67755,1.65002,1.62893,1.61322,1.60194,
    #       1.5943,1.58965,1.58964,1.5872,0.]

    numSubDomains = np.prod(subDomains)

    drain = True
    test = False

    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,periodic,inlet,outlet,"InkBotle",None,None)
    sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid)
    if drain:
        drainL = PMMoTo.calcDrainage(rank,size,pC,domain,sDL,inlet,sDEDTL)


# pC = [7.69409,7.69409,7.69393,7.60403,7.36005,6.9909,6.53538,6.03352,5.52008,5.02123,
# 4.55421,4.12854,3.74806,3.41274,3.12024,2.86704,2.64914,2.4625,2.30332,2.16814,2.05388,
# 1.95783,1.87764,1.81122,1.75678,1.7127,1.67755,1.65002,1.62893,1.61322,1.60194,
# 1.5943,1.58965,1.58964,1.5872,0.]
#
# sWAnalytical = [0.,0.808933,0.808965,0.809662,0.810321,0.810973,0.811641,0.81235,0.813122,
#                 0.813978,0.814941,0.816034,0.817281,0.818703,0.820323,0.822162,0.824238,
#                 0.826565,0.829154,0.832009,0.835131,0.838511,0.842134,0.845978,0.850016,
#                 0.854214,0.858539,0.862957,0.867443,0.871985,0.876591,0.881302,0.886197,1.,1.,1.]
#




    if test:
        if rank == 0:
            sD = np.empty((numSubDomains), dtype = object)
            sDEDT = np.empty((numSubDomains), dtype = object)
            sD[0] = sDL
            sDEDT[0] = sDEDTL
            if drain:
                sDdrain = np.empty((numSubDomains), dtype = object)
                sDdrain[0] = drainL
            for neigh in range(1,numSubDomains):
                sD[neigh] = comm.recv(source=neigh)
                sDEDT[neigh] = comm.recv(source=neigh)
                if drain:
                    sDdrain[neigh] = comm.recv(source=neigh)

        if rank > 0:
            for neigh in range(1,numSubDomains):
                if rank == neigh:
                    comm.send(sDL,dest=0)
                    comm.send(sDEDTL,dest=0)
                    if drain:
                        comm.send(drainL,dest=0)


        if rank ==0:

            ## Gen Single Domain
            x = np.linspace(domain.domainSize[0,0] + domain.dX/2, domain.domainSize[0,1]-domain.dX/2, nodes[0])
            y = np.linspace(domain.domainSize[1,0] + domain.dY/2, domain.domainSize[1,1]-domain.dY/2, nodes[1])
            z = np.linspace(domain.domainSize[2,0] + domain.dZ/2, domain.domainSize[2,1]-domain.dZ/2, nodes[2])
            gridOut = np.zeros(nodes,dtype=np.int8)
            for i in range(0,nodes[0]):
                for j in range(0,nodes[1]):
                    for k in range(0,nodes[2]):
                        r = (0.01*math.cos(0.01*x[i])+0.5*math.sin(x[i])+0.75)
                        if (z[k]*z[k] + y[j]*y[j]) <= r*r:
                            gridOut[i,j,k]=1

            realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
            edtV,indTrue = distance_transform_edt(gridOut,sampling=[domain.dX, domain.dY, domain.dZ],return_indices=True)

            ### Reconstruct SubDomains to Check EDT ####
            checkGrid = np.zeros_like(realDT)
            n = 0
            for i in range(0,subDomains[0]):
                for j in range(0,subDomains[1]):
                    for k in range(0,subDomains[2]):
                        checkGrid[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
                                  sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
                                  sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
                                  = sD[n].grid[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
                                                  sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
                                                  sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
                        n = n + 1

            diffGrid = gridOut-checkGrid
            print("L2 Grid Error Norm",np.linalg.norm(diffGrid) )


            checkEDT = np.zeros_like(realDT)
            n = 0
            for i in range(0,subDomains[0]):
                for j in range(0,subDomains[1]):
                    for k in range(0,subDomains[2]):
                        checkEDT[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
                                 sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
                                 sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
                                 = sDEDT[n].EDT[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
                                                sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
                                                sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
                        n = n + 1

            diffEDT = np.abs(realDT-checkEDT)
            diffEDT2 = np.abs(edtV-checkEDT)
            print("L2 EDT Error Norm",np.linalg.norm(diffEDT) )
            print("L2 EDT Error Norm 2",np.linalg.norm(diffEDT2) )
            print("L2 EDT Error Norm 2",np.linalg.norm(realDT-edtV) )

            print("LI EDT Error Norm",np.max(diffEDT) )
            print("LI EDT Error Norm 2",np.max(diffEDT2) )
            print("LI EDT Error Norm 2",np.max(realDT-edtV) )


            # pdb.set_trace()
            # for nn in range(0,numSubDomains):
            #     numSets = sDdrain[nn].setCount
            #     totalNodes = 0
            #     for n in range(0,numSets):
            #         totalNodes = totalNodes + len(sDdrain[nn].Sets[n].nodes)
            #     printSetOut = np.zeros([totalNodes,7])
            #     c = 0
            #     for setID in range(0,numSets):
            #         for n in range(0,len(sDdrain[nn].Sets[setID].nodes)):
            #             printSetOut[c,0] = sD[nn].indexStart[0] + sDdrain[nn].Sets[setID].nodes[n].localIndex[0]
            #             printSetOut[c,1] = sD[nn].indexStart[1] + sDdrain[nn].Sets[setID].nodes[n].localIndex[1]
            #             printSetOut[c,2] = sD[nn].indexStart[2] + sDdrain[nn].Sets[setID].nodes[n].localIndex[2]
            #             printSetOut[c,3] = sDdrain[nn].Sets[setID].globalID
            #             printSetOut[c,4] = nn
            #             printSetOut[c,5] = sDdrain[nn].Sets[setID].inlet
            #             printSetOut[c,6] = sDdrain[nn].Sets[setID].nodes[n].dist
            #             c = c + 1
            #         #print(nn,allDrainage[nn].Sets[setID].globalID)
            #     file = "dataDump/3dsubGridSetNodes_"+str(nn)+".csv"
            #     header = "x,y,z,globalID,procID,Inlet,Dist"
            #     np.savetxt(file,printSetOut, delimiter=',',header=header)

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
            #             printGridOut[c,5] = diffEDT[i,j,k]
            #             c = c + 1
            #
            # header = "x,y,z,Grid,EDT,diffEDT,No"
            # file = "dataDump/3dGridOut.csv"
            # np.savetxt(file,printGridOut, delimiter=',',header=header)
            #
            #

            for nn in range(0,numSubDomains):
                printGridOut = np.zeros([sD[nn].grid.size,8])
                c = 0
                for i in range(0,sD[nn].grid.shape[0]):
                    for j in range(0,sD[nn].grid.shape[1]):
                        for k in range(0,sD[nn].grid.shape[2]):
                            printGridOut[c,0] = sD[nn].x[i]#sDAll[nn].indexStart[0] + i #sDAll[nn].x[i]
                            printGridOut[c,1] = sD[nn].y[j]#sDAll[nn].indexStart[1] + j#sDAll[nn].y[j]
                            printGridOut[c,2] = sD[nn].z[k]#sDAll[nn].indexStart[2] + k#sDAll[nn].z[k]
                            printGridOut[c,3] = sD[nn].grid[i,j,k]
                            printGridOut[c,4] = sDEDT[nn].EDT[i,j,k]
                            printGridOut[c,5] = sDdrain[nn].nwp[i,j,k]
                            printGridOut[c,6] = sDdrain[nn].nwpFinal[i,j,k]
                            printGridOut[c,7] = nn
                            c = c + 1

                header = "x,y,z,Grid,EDT,NWP,NWPFinal,ProcID"
                file = "dataDump/3dsubGridOut_"+str(nn)+".csv"
                np.savetxt(file,printGridOut, delimiter=',',header=header)

            # nn = 0
            # c = 0
            # data = np.empty((0,3))
            # orient = sD[0].Orientation
            # for fIndex in orient.faces:
            #     #orientID = orient.faces[fIndex]['ID']
            #     #for procs in sDEDT[nn].solidsAll.keys():
            #         #for fID in sDEDT[nn].solidsAll[procs]['orientID'].keys():
            #             #if fID == orientID:
            #     data = np.append(data,sDEDT[nn].trimmedSolids[fIndex],axis=0)
            #                 #data = np.append(data,sDEDT[nn].solidsAll[procs]['orientID'][fID],axis=0)
            # printGridOut = np.zeros([data.shape[0],3])
            # dX = 0.05
            # dY = 0.05
            # dZ = 0.05
            # print(data.shape[0])
            # for i in range(data.shape[0]):
            #     printGridOut[c,0] = data[i,0]*dX
            #     printGridOut[c,1] = data[i,1]*dY - 1.5
            #     printGridOut[c,2] = data[i,2]*dZ - 1.5
            #     # printGridOut[c,0] = data[i,0]
            #     # printGridOut[c,1] = data[i,1]
            #     # printGridOut[c,2] = data[i,2]
            #     c = c + 1
            #
            # header = "x,y,z"
            # file = "dataDump/3dsubGridSINGLEAllSolids_"+str(nn)+".csv"
            # np.savetxt(file,printGridOut, delimiter=',',header=header)



if __name__ == "__main__":
    my_function()
