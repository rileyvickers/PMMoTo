import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
import os
import edt
import time
import PMMoTo

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
    #nodes = [928,928,1340]
    #nodes = [696,696,1005]
    #nodes = [464,464,670]
    #nodes = [232,232,335]
    #nodes = [116,116,168]
    nodes = [51,51,51]
    boundaries = [0,0,0]
    inlet  = [1,0,0]
    outlet = [-1,0,0]
    file = './testDomains/50pack.out'
    # file = './testDomains/pack_sub.dump.gz'
    #domainFile = open('kelseySpherePackTests/pack_res.out', 'r')
    res = 1 ### Assume that the reservoir is always at the inlet!

    numSubDomains = np.prod(subDomains)


    drain = False
    testSerial = True

    pC = [143]

    startTime = time.time()

    # domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,res,"Sphere",file,PMMoTo.readPorousMediaLammpsDump)
    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,res,"Sphere",file,PMMoTo.readPorousMediaXYZR)

    sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid,stats = True)

    if drain:
        drainL,_ = PMMoTo.calcDrainage(rank,size,pC,domain,sDL,inlet,sDEDTL)

    rad = 0.1
    sDMorphL = PMMoTo.morph(rank,size,domain,sDL,sDL.grid,rad)
    sDMAL = PMMoTo.medialAxis.medialAxisEval(rank,size,domain,sDL,sDL.grid)

    endTime = time.time()

    print("Parallel Time:",endTime-startTime)

    if testSerial:

        if rank == 0:
            sD = np.empty((numSubDomains), dtype = object)
            sDEDT = np.empty((numSubDomains), dtype = object)
            if drain:
                sDDrain = np.empty((numSubDomains), dtype = object)
            sDMorph = np.empty((numSubDomains), dtype = object)
            sDMA = np.empty((numSubDomains), dtype = object)
            sD[0] = sDL
            sDEDT[0] = sDEDTL
            if drain:
                sDDrain[0] = drainL
            sDMorph[0] = sDMorphL
            sDMA[0] = sDMAL
            for neigh in range(1,numSubDomains):
                sD[neigh] = comm.recv(source=neigh)
                sDEDT[neigh] = comm.recv(source=neigh)
                if drain:
                    sDDrain[neigh] = comm.recv(source=neigh)
                sDMorph[neigh] = comm.recv(source=neigh)
                sDMA[neigh] = comm.recv(source=neigh)

        if rank > 0:
            for neigh in range(1,numSubDomains):
                if rank == neigh:
                    comm.send(sDL,dest=0)
                    comm.send(sDEDTL,dest=0)
                    if drain:
                        comm.send(drainL,dest=0)
                    comm.send(sDMorphL,dest=0)
                    comm.send(sDMAL,dest=0)


        if rank==0:
            testAlgo = True
            if testAlgo:
                
                if not os.path.exists('dataDump'):
                    os.mkdir('dataDump')
                startTime = time.time()

                _,sphereData = PMMoTo.readPorousMediaXYZR(file)
                # domainSize,sphereData = PMMoTo.readPorousMediaLammpsDump(file)

                ##### To GENERATE SINGLE PROC TEST CASE ######
                x = np.linspace(domain.dX/2, domain.domainSize[0,1]-domain.dX/2, nodes[0])
                y = np.linspace(domain.dY/2, domain.domainSize[1,1]-domain.dY/2, nodes[1])
                z = np.linspace(domain.dZ/2, domain.domainSize[2,1]-domain.dZ/2, nodes[2])

                gridOut = PMMoTo.domainGen(x,y,z,sphereData)
                gridOut = np.asarray(gridOut)

                pG = [0,0,0]
                pgSize = nodes[0]

                if boundaries[0] == 1:
                    pG[0] = 1
                if boundaries[1] == 1:
                    pG[1] = 1
                if boundaries[2] == 1:
                    pG[2] = 1

                periodic = [False,False,False]
                if boundaries[0] == 2:
                    periodic[0] = True
                    pG[0] = pgSize
                if boundaries[1] == 2:
                    periodic[1] = True
                    pG[1] = pgSize
                if boundaries[2] == 2:
                    periodic[2] = True
                    pG[2] = pgSize

                gridOut = np.pad (gridOut, ((pG[0], pG[0]), (pG[1], pG[1]), (pG[2], pG[2])), 'wrap')

                print(pG)

                if boundaries[0] == 1:
                    gridOut[0,:,:] = 0
                    gridOut[-1,:,:] = 0
                if boundaries[1] == 1:
                    gridOut[:,0,:] = 0
                    gridOut[:,-1,:] = 0
                if boundaries[2] == 1:
                    gridOut[:,:,0] = 0
                    gridOut[:,:,-1] = 0


                realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
                edtV,indTrue = distance_transform_edt(gridOut,sampling=[domain.dX, domain.dY, domain.dZ],return_indices=True)
                gridCopy = np.copy(gridOut)
                realMA = PMMoTo.medialAxis.skeletonize._compute_thin_image(gridCopy)
                endTime = time.time()

                print("Serial Time:",endTime-startTime)

                if pG[0] > 0 and pG[1]==0 and pG[2]==0:
                    gridOut = gridOut[pG[0]:-pG[0],:,:]
                    realDT = realDT[pG[0]:-pG[0],:,:]
                    edtV = edtV[pG[0]:-pG[0],:,:]
                    realMA = realMA[pG[0]:-pG[0],:,:]

                elif pG[0]==0 and pG[1] > 0 and pG[2]==0:
                    gridOut = gridOut[:,pG[1]:-pG[1],:]
                    realDT = realDT[:,pG[1]:-pG[1],:]
                    edtV = edtV[:,pG[1]:-pG[1],:]
                    realMA = realMA[:,pG[1]:-pG[1],:]

                elif pG[0]==0 and pG[1]==0 and pG[2] > 0:
                    gridOut = gridOut[:,:,pG[2]:-pG[2]]
                    realDT = realDT[:,:,pG[2]:-pG[2]]
                    edtV = edtV[:,:,pG[2]:-pG[2]]
                    realMA = realMA[:,:,pG[2]:-pG[2]]

                elif pG[0] > 0 and pG[1]==0 and pG[2] > 0:
                    gridOut = gridOut[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    realDT = realDT[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    edtV = edtV[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    realMA = realMA[pG[0]:-pG[0],:,pG[2]:-pG[2]]

                elif pG[0] > 0 and pG[1] > 0 and pG[2]==0:
                    gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    realMA = realMA[pG[0]:-pG[0],pG[1]:-pG[1],:]

                elif pG[0]==0 and pG[1] > 0 and pG[2] > 0:
                    gridOut = gridOut[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    realDT = realDT[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    edtV = edtV[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    realMA = realMA[:,pG[1]:-pG[1],pG[2]:-pG[2]]

                elif pG[0] > 0 and pG[1] > 0 and pG[2] > 0:
                    gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    realMA = realMA[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                ####################################################################


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

                checkMA = np.zeros_like(realDT)
                n = 0
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
                            checkMA[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
                                     sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
                                     sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
                                     = sDMA[n].MA[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
                                                    sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
                                                    sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
                            n = n + 1

                diffMA = np.abs(realMA-checkMA)
                print(realMA.shape,checkMA.shape)
                print("L2 MA Error Total Different Voxels",np.sum(diffMA) )


                c = 0
                printGridOut = np.zeros([gridOut.size,5])
                for i in range(0,gridOut.shape[0]):
                    for j in range(0,gridOut.shape[1]):
                        for k in range(0,gridOut.shape[2]):
                            printGridOut[c,0] = i#x[i]
                            printGridOut[c,1] = j#y[j]
                            printGridOut[c,2] = k#z[k]
                            ci = i 
                            cj = j 
                            ck = k
                            printGridOut[c,3] = gridOut[ci,cj,ck]
                            printGridOut[c,4] = realMA[ci,cj,ck]
                            #printGridOut[c,4] = checkMA[ci,cj,ck]
                            #printGridOut[c,5] = gridOut[ci,cj,ck]
                            c = c + 1
                
                header = "x,y,z,RealMA,CheckMA,GRID"#,Grid,Dist"
                header = "i,j,k,Grid,MA"#,Grid,Dist"
                file = "dataDump/3GridWALLS.csv"
                np.savetxt(file,printGridOut, delimiter=',',header=header)
                #
                # c = 0
                # printGridOut = np.zeros([gridOut.size,7])
                # for i in range(0,gridOut.shape[0]):
                #     for j in range(0,gridOut.shape[1]):
                #         for k in range(0,gridOut.shape[2]):
                #             printGridOut[c,0] = x[i]
                #             printGridOut[c,1] = y[j]
                #             printGridOut[c,2] = z[k]
                #             printGridOut[c,3] = realMA[i,j,k]
                #             printGridOut[c,4] = checkMA[i,j,k]
                #             printGridOut[c,5] = gridOut[i,j,k]
                #             printGridOut[c,6] = realDT[i,j,k]
                #             c = c + 1
                #
                # header = "x,y,z,RealMA,CheckMA,GRID,DIST"
                # file = "dataDump/3GridORIG.csv"
                # np.savetxt(file,printGridOut, delimiter=',',header=header)


            # for nn in range(0,numSubDomains):
            #     #numSum = np.sum(sD[nn].grid.size)
            #     #printGridOut = np.zeros([sD[nn].grid.shape[1]*sD[nn].grid.shape[2]*2,4])
            #     #printGridOut = np.zeros([sDMA[nn].haloGrid.size,7])
            #     printGridOut = np.zeros([np.sum(sDMA[nn].MA),7])
            #     c = 0
            #     print(sDMA[nn].halo)
            #     print(nn,sD[nn].indexStart)
            #     for i in range(-sDMA[nn].halo[1],sDMA[nn].haloGrid.shape[0]-sDMA[nn].halo[1]):
            #         for j in range(-sDMA[nn].halo[3],sDMA[nn].haloGrid.shape[1]-sDMA[nn].halo[3]):
            #             for k in range(-sDMA[nn].halo[5],sDMA[nn].haloGrid.shape[2]-sDMA[nn].halo[5]):
            #                 #if sD[nn].grid[i,j,k] == 1:
            #                 ci = i+sDMA[nn].halo[1]
            #                 cj = j+sDMA[nn].halo[3]
            #                 ck = k+sDMA[nn].halo[5]
            #                 ipA = False; jpA = False; kpA = False;
            #                 if i > -5 and i < sDMA[nn].haloGrid.shape[0]-sDMA[nn].halo[1]-sDMA[nn].halo[0] + 5:
            #                     ipA = True
            #                 if j > -5 and j < sDMA[nn].haloGrid.shape[1]-sDMA[nn].halo[3]-sDMA[nn].halo[2] + 5:
            #                     jpA = True
            #                 if k > -5 and k < sDMA[nn].haloGrid.shape[2]-sDMA[nn].halo[5]-sDMA[nn].halo[4] + 5:
            #                     kpA = True
            #                 if sDMA[nn].MA[ci,cj,ck] == 1 and ipA and jpA and kpA:
            #                     # ci = i+sDMA[nn].halo[1]
            #                     # cj = j+sDMA[nn].halo[3]
            #                     # ck = k+sDMA[nn].halo[5]
            #                     printGridOut[c,0] = sD[nn].indexStart[0]- sDMA[nn].halo[1] + ci #sDAll[nn].x[i]
            #                     printGridOut[c,1] = sD[nn].indexStart[1] - sDMA[nn].halo[3] + cj#sDAll[nn].y[j]
            #                     printGridOut[c,2] = sD[nn].indexStart[2] - sDMA[nn].halo[5] + ck#sDAll[nn].z[k]
            #                     printGridOut[c,3] = sDMA[nn].MA[ci,cj,ck]
            #                     printGridOut[c,4] = sDMA[nn].haloGrid[ci,cj,ck]
            #                     printGridOut[c,5] = nn
            #                     # printGridOut[c,6] = sDdrain[nn].nwp[i,j,k]
            #                     c = c + 1
            #
            #     header = "x,y,z,MA,GRID,ProcID"#,Grid,Dist"
            #     file = "dataDump/3dsubHaloGrid_"+str(nn)+".csv"
            #     np.savetxt(file,printGridOut, delimiter=',',header=header)
            #
            for nn in range(0,numSubDomains):
                printGridOut = np.zeros([sD[nn].grid.size,8])
                c = 0
                for i in range(0,sD[nn].grid.shape[0]):
                    for j in range(0,sD[nn].grid.shape[1]):
                        for k in range(0,sD[nn].grid.shape[2]):
                            #if sD[nn].grid[i,j,k] == 1:
                                printGridOut[c,0] = sD[nn].indexStart[0] + i #sDAll[nn].x[i]
                                printGridOut[c,1] = sD[nn].indexStart[1] + j #sDAll[nn].y[j]
                                printGridOut[c,2] = sD[nn].indexStart[2] + k #sDAll[nn].z[k]
                                printGridOut[c,3] = sD[nn].grid[i,j,k]
                                printGridOut[c,4] = sDMA[nn].MA[i,j,k]
                                printGridOut[c,5] = nn
                                cID = sDMA[nn].nodeTable[i,j,k]
                                printGridOut[c,6] = sDMA[nn].nodeInfoIndex[cID,3]
                                printGridOut[c,7] = cID
                                c = c + 1

                header = "x,y,z,GRID,MA,ProcID,globINDEX,cID"#,Grid,Dist"
                file = "dataDump/3dsubGrid_"+str(nn)+".csv"
                np.savetxt(file,printGridOut, delimiter=',',header=header)


            for nn in range(0,numSubDomains):
                c = 0
                printGridOut = np.zeros([np.sum(sDMA[nn].MA),12])
                for ss in range(0,sDMA[nn].setCount):
                    #print(nn,ss,sDMA[nn].Sets[ss].localID,sDMA[nn].Sets[ss].globalID)
                    for no in sDMA[nn].Sets[ss].nodes:
                        printGridOut[c,0] = sD[nn].indexStart[0]+no[0]
                        printGridOut[c,1] = sD[nn].indexStart[1]+no[1]
                        printGridOut[c,2] = sD[nn].indexStart[2]+no[2]
                        printGridOut[c,3] = sDMA[nn].Sets[ss].localID
                        printGridOut[c,4] = nn
                        printGridOut[c,5] = sDMA[nn].Sets[ss].type
                        cID = sDMA[nn].nodeTable[no[0],no[1],no[2]]
                        printGridOut[c,6] = cID
                        printGridOut[c,7] = sDMA[nn].Sets[ss].pathID
                        printGridOut[c,8] = sDMA[nn].Sets[ss].globalID
                        printGridOut[c,9] = sDMA[nn].Sets[ss].inlet
                        printGridOut[c,10] = sDMA[nn].Sets[ss].outlet
                        printGridOut[c,11] = sDEDT[nn].EDT[no[0],no[1],no[2]]
                        c = c + 1

                header = "x,y,z,SetID,ProcID,TYPE,ID,pathID,globalID,Inlet,Outlet,Dist"#,Grid,Dist"
                file = "dataDump/3dsubGridMA_"+str(nn)+".csv"
                np.savetxt(file,printGridOut, delimiter=',',header=header)


                #
                # for nn in range(0,numSubDomains):
                #     numSets = sDdrain[nn].setCount
                #     totalNodes = 0
                #     for n in range(0,numSets):
                #         totalNodes = totalNodes + len(sDdrain[nn].Sets[n].boundaryNodes)
                #     printSetOut = np.zeros([totalNodes,7])
                #     c = 0
                #     for setID in range(0,numSets):
                #         for n in range(0,len(sDdrain[nn].Sets[setID].boundaryNodes)):
                #             printSetOut[c,0] = sDdrain[nn].Sets[setID].boundaryNodeID[n,0]
                #             printSetOut[c,1] = sDdrain[nn].Sets[setID].boundaryNodeID[n,1]
                #             printSetOut[c,2] = sDdrain[nn].Sets[setID].boundaryNodeID[n,2]
                #             printSetOut[c,3] = sDdrain[nn].Sets[setID].globalID
                #             printSetOut[c,4] = nn
                #             printSetOut[c,5] = sDdrain[nn].Sets[setID].inlet
                #             printSetOut[c,6] = setID
                #             c = c + 1
                #         #print(nn,allDrainage[nn].Sets[setID].globalID)
                #     file = "dataDump/3dsubGridSetNodes_"+str(nn)+".csv"
                #     header = "x,y,z,globalID,procID,Inlet,SetID"
                #     np.savetxt(file,printSetOut, delimiter=',',header=header)


                #printGridOut = np.zeros([grid.size,5])
                # c = 0
                # for i in range(0,grid.shape[0]):
                #     for j in range(0,grid.shape[1]):
                #         for k in range(0,grid.shape[2]):
                #             if diffEDT[i,j,k] > 1.e-6:
                #                 print(i,j,k,x[i],y[j],z[k],diffEDT[i,j,k],checkEDT[i,j,k],realDT[i,j,k],indTrue[0][i,j,k],indTrue[1][i,j,k],indTrue[2][i,j,k])
                #                 # printGridOut[c,0] = x[i]
                #                 # printGridOut[c,1] = y[j]
                #                 # printGridOut[c,2] = z[k]
                #                 # printGridOut[c,3] = gridOut[i,j,k]
                #                 # printGridOut[c,4] = diffEDT[i,j,k]
                #                 c = c + 1

                # header = "x,y,z,Grid,EDT"
                # file = "dataDump/3dGrid.csv"
                # np.savetxt(file,printGridOut, delimiter=',',header=header)
            #
            # for nn in range(0,numSubDomains):
            #     printGridOut = np.zeros([sD[nn].grid.size,5])
            #     c = 0
            #     for i in range(0,sD[nn].grid.shape[0]):
            #         for j in range(0,sD[nn].grid.shape[1]):
            #             for k in range(0,sD[nn].grid.shape[2]):
            #                 if sD[nn].grid[i,j,k] == 1:
            #                     printGridOut[c,0] = sD[nn].x[i]#sDAll[nn].indexStart[0] + i #sDAll[nn].x[i]
            #                     printGridOut[c,1] = sD[nn].y[j]#sDAll[nn].indexStart[1] + j#sDAll[nn].y[j]
            #                     printGridOut[c,2] = sD[nn].z[k]#sDAll[nn].indexStart[2] + k#sDAll[nn].z[k]
            #                     printGridOut[c,3] = sDEDT[nn].EDT[i,j,k]
            #                     printGridOut[c,4] = sD[nn].grid[i,j,k]
            #                     c = c + 1
            #
            #     header = "x,y,z,DiffEDT,EDT"
            #     file = "dataDump/3dsubGrid_"+str(nn)+".csv"
            #     np.savetxt(file,printGridOut, delimiter=',',header=header)
            #
            #
            # nn = 0
            # c = 0
            # data = np.empty((0,3))
            # orient = sDL.Orientation
            # for fIndex in orient.faces:
            #     orientID = orient.faces[fIndex]['ID']
            #     for procs in sDEDT[nn].solidsAll.keys():
            #         for fID in sDEDT[nn].solidsAll[procs]['orientID'].keys():
            #             if fID == orientID:
            #                 data = np.append(data,sDEDT[nn].solidsAll[procs]['orientID'][fID],axis=0)
            # printGridOut = np.zeros([data.shape[0],5])
            # for i in range(data.shape[0]):
            #     printGridOut[c,0] = data[i][0]
            #     printGridOut[c,1] = data[i][1]
            #     printGridOut[c,2] = data[i][2]
            #     c = c + 1
            #
            # header = "x,y,z"
            # file = "dataDump/3dsubGridAllSolids_"+str(nn)+".csv"
            # np.savetxt(file,printGridOut, delimiter=',',header=header)


if __name__ == "__main__":
    my_function()
    #MPI.Finalize()
