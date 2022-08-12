import numpy as np
from mpi4py import MPI
from scipy import ndimage
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
import pdb
import edt
import sys
import time
import PMMoTo
import math

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
    nodes = [232,232,335]
    #nodes = [116,116,168]
    periodic = [True,True,False]
    inlet  = [0,0,-1]
    outlet = [0,0, 1]
    #domainFile = open('testDomains/pack_sub.out', 'r')
    domainFile = open('kelseySpherePackTests/pack_res.out', 'r')

    numSubDomains = np.prod(subDomains)


    drain = True
    testSerial = False

    #pC = [5.295714695]
    #pC = [10.85910666]
    pC = [11.43063876]
    #pC = [1,2,3]
    #pC = [100]
    #pC = [75,100,125,150,175,200]

    #pC = [3.35,3.53,3.91,8.02,11.48,12.08,12.72,13.39,14.84,17.31,21.25,30.42]
    # pC = [1,2,3,3.45653767,3.638455891,4.031533718,5,6,7,8,8.266844475,8.5,9,9.5,10,10.5,11,11.5,11.83790489,12.4609337,
    #       13.11679317,13.80715071,14,15,15.29872491,16,17.84376896,18,19,20,21.90730472,31.37079913,35,40,45,50,75,100,125,150,175,200]
    #pC = np.linspace(1,25,96*4+1)


    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,periodic,inlet,outlet,"Sphere",domainFile,PMMoTo.readPorousMediaXYZR)
    sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid,stats = True)
    if drain:
        drainL = PMMoTo.calcDrainage(rank,size,pC,domain,sDL,inlet,sDEDTL,info=False)



    #### Generate Slice
    slice = 0 # 0 = x, 1= y, 2 = z
    sliceLoc = domain.domainSize[0,1]/2

    if slice == 0:
        sliceLow  = sliceLoc - domain.dX
        sliceHigh = sliceLoc + domain.dX
        indID = np.where( (sDL.x > sliceLow) & (sDL.x < sliceHigh) )[0]
        totalNodes = indID.size*(sDL.ownNodes[1][1]-sDL.ownNodes[1][0])*(sDL.ownNodes[2][1]-sDL.ownNodes[2][0])
        printGridOut = np.zeros([totalNodes,4])

        c = 0
        for i in indID:
            for j in range(sDL.ownNodes[1][0],sDL.ownNodes[1][1]):
                for k in range(sDL.ownNodes[2][0],sDL.ownNodes[2][1]):
                    printGridOut[c,0] = sDL.x[i]
                    printGridOut[c,1] = sDL.y[j]
                    printGridOut[c,2] = sDL.z[k]
                    printGridOut[c,3] = drainL.nwpFinal[i,j,k]
                    c = c + 1
        header = "x,y,z,NWPFinal"
        file = "dataDump/3dSlice_"+str(rank)+".csv"
        np.savetxt(file,printGridOut, delimiter=',',header=header)

    elif slice == 1:
        sliceLow  = sliceLoc - domain.dY
        sliceHigh = sliceLoc + domain.dY
        indID = np.where( (sDL.y > sliceLow) & (sDL.y < sliceHigh) )

        totalNodes = indID.size*(sDL.ownNodes[0][1]-sDL.ownNodes[0][0])*(sDL.ownNodes[2][1]-sDL.ownNodes[2][0])
        printGridOut = np.zeros([totalNodes,4])

        c = 0
        for i in range(sDL.ownNodes[0][0],sDL.ownNodes[0][1]):
            for j in indID:
                for k in range(sDL.ownNodes[2][0],sDL.ownNodes[2][1]):
                    printGridOut[c,0] = sDL.x[i]
                    printGridOut[c,1] = sDL.y[j]
                    printGridOut[c,2] = sDL.z[k]
                    printGridOut[c,3] = drainL.nwpFinal[i,j,k]
                    c = c + 1
        header = "x,y,z,NWPFinal"
        file = "dataDump/3dSlice_"+str(rank)+".csv"
        np.savetxt(file,printGridOut, delimiter=',',header=header)


    elif slice == 2:
        sliceLow  = sliceLoc - domain.dZ
        sliceHigh = sliceLoc + domain.dZ
        indID = np.where( (sDL.z > sliceLow) & (sDL.z < sliceHigh) )

        totalNodes = indID.size*(sDL.ownNodes[0][1]-sDL.ownNodes[0][0])*(sDL.ownNodes[1][1]-sDL.ownNodes[1][0])
        printGridOut = np.zeros([totalNodes,4])

        c = 0
        for i in range(sDL.ownNodes[0][0],sDL.ownNodes[0][1]):
            for j in range(sDL.ownNodes[1][0],sDL.ownNodes[1][1]):
                for k in indID:
                    printGridOut[c,0] = sDL.x[i]
                    printGridOut[c,1] = sDL.y[j]
                    printGridOut[c,2] = sDL.z[k]
                    printGridOut[c,3] = drainL.nwpFinal[i,j,k]
                    c = c + 1
        header = "x,y,z,NWPFinal"
        file = "dataDump/3dSlice_"+str(rank)+".csv"
        np.savetxt(file,printGridOut, delimiter=',',header=header)




    if testSerial:

        #### TESTING and PLOTTING ####
        if rank == 0:
            print("--- %s seconds ---" % (time.time() - start_time))
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


        if rank==0:
            testAlgo = False
            if testAlgo:

                startTime = time.time()

                domainFile.close()
                #domainFile = open('testDomains/pack_sub.out', 'r')
                domainFile = open('kelseySpherePackTests/pack_res.out', 'r')
                domainSize,sphereData = PMMoTo.readPorousMediaXYZR(domainFile)

                ##### To GENERATE SINGLE PROC TEST CASE ######
                x = np.linspace(domain.dX/2, domain.domainSize[0,1]-domain.dX/2, nodes[0])
                y = np.linspace(domain.dY/2, domain.domainSize[1,1]-domain.dY/2, nodes[1])
                z = np.linspace(domain.dZ/2, domain.domainSize[2,1]-domain.dZ/2, nodes[2])
                xyz = np.meshgrid(x,y,z,indexing='ij')

                grid = np.ones([nodes[0],nodes[1],nodes[2]],dtype=np.integer)
                gridOut = PMMoTo.domainGen(x,y,z,sphereData)
                gridOut = np.asarray(gridOut)

                pG = [0,0,0]
                if periodic[0] == True:
                    pG[0] = 50
                if periodic[1] == True:
                    pG[1] = 50
                if periodic[2] == True:
                    pG[2] = 50


                gridOut = np.pad (gridOut, ((pG[0], pG[0]), (pG[1], pG[1]), (pG[2], pG[2])), 'wrap')
                realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
                edtV,indTrue = distance_transform_edt(gridOut,sampling=[domain.dX, domain.dY, domain.dZ],return_indices=True)
                endTime = time.time()

                print("Serial Time:",endTime-startTime)


                printDISTANCEOut = np.zeros([realDT.shape[0]*realDT.shape[1]*realDT.shape[2]*2,4])
                c = 0
                for i in range(0,realDT.shape[0]):
                    for j in range(0,realDT.shape[1]):
                        for k in range(0,realDT.shape[2]):
                            printDISTANCEOut[c,0] = i
                            printDISTANCEOut[c,1] = j
                            printDISTANCEOut[c,2] = k
                            if gridOut[i,j,k] == 1:
                                printDISTANCEOut[c,3] = realDT[i,j,k]
                            else:
                                printDISTANCEOut[c,3] = -1
                            c = c + 1
                header = "X,Y,Z,EDT"
                file = "dataDump/Distance.csv"
                np.savetxt(file,printDISTANCEOut, delimiter=',',header=header)

                if periodic[0] and not periodic[1] and not periodic[2]:
                    gridOut = gridOut[pG[0]:-pG[0],:,:]
                    realDT = realDT[pG[0]:-pG[0],:,:]
                    edtV = edtV[pG[0]:-pG[0],:,:]
                elif not periodic[0] and periodic[1] and not periodic[2]:
                    gridOut = gridOut[:,pG[1]:-pG[1],:]
                    realDT = realDT[:,pG[1]:-pG[1],:]
                    edtV = edtV[:,pG[1]:-pG[1],:]
                elif not periodic[0] and not periodic[1] and periodic[2]:
                    gridOut = gridOut[:,:,pG[2]:-pG[2]]
                    realDT = realDT[:,:,pG[2]:-pG[2]]
                    edtV = edtV[:,:,pG[2]:-pG[2]]
                elif periodic[0] and not periodic[1] and periodic[2]:
                    gridOut = gridOut[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    realDT = realDT[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    edtV = edtV[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                elif periodic[0] and periodic[1] and not periodic[2]:
                    print("HELLoO PERIODIC")
                    gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],:]
                elif not periodic[0] and periodic[1] and periodic[2]:
                    gridOut = gridOut[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    realDT = realDT[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    edtV = edtV[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                elif periodic[0] and periodic[1] and periodic[2]:
                    gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
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

                # printDISTANCEOut = np.zeros([checkEDT.shape[0]*checkEDT.shape[1]*checkEDT.shape[2]*2,1])
                # c = 0
                # for i in range(0,checkEDT.shape[0]):
                #     for j in range(0,checkEDT.shape[1]):
                #         for k in range(0,checkEDT.shape[2]):
                #             # printDISTANCEOut[c,0] = i
                #             # printDISTANCEOut[c,1] = j
                #             # printDISTANCEOut[c,2] = k
                #             if gridOut[i,j,k] == 1:
                #                 printDISTANCEOut[c,0] = realDT[i,j,k]
                #             else:
                #                 printDISTANCEOut[c,0] = -1
                #             c = c + 1
                # header = "EDT"
                # file = "dataDump/Distance.csv"
                # np.savetxt(file,printDISTANCEOut, delimiter=',',header=header)
                # file = "dataDump/Distance.npy"
                # np.savetxt(file,printDISTANCEOut)





            for nn in range(0,numSubDomains):
                #numSum = np.sum(sD[nn].grid.size)
                #printGridOut = np.zeros([sD[nn].grid.shape[1]*sD[nn].grid.shape[2]*2,4])
                printGridOut = np.zeros([sD[nn].grid.size,6])
                c = 0
                for i in range(0,sD[nn].grid.shape[0]):
                    for j in range(0,sD[nn].grid.shape[1]):
                        for k in range(0,sD[nn].grid.shape[2]):
                            #if sD[nn].grid[i,j,k] == 1:
                                printGridOut[c,0] = sD[nn].x[i]#sDAll[nn].indexStart[0] + i #sDAll[nn].x[i]
                                printGridOut[c,1] = sD[nn].y[j]#sDAll[nn].indexStart[1] + j#sDAll[nn].y[j]
                                printGridOut[c,2] = sD[nn].z[k]#sDAll[nn].indexStart[2] + k#sDAll[nn].z[k]
                                printGridOut[c,3] = sDdrain[nn].nwpFinal[i,j,k]
                                # printGridOut[c,4] = sD[nn].grid[i,j,k]
                                # printGridOut[c,5] = sDEDT[nn].EDT[i,j,k]
                                # printGridOut[c,6] = sDdrain[nn].nwp[i,j,k]
                                c = c + 1

                header = "x,y,z,NWPFinal"#,Grid,Dist"
                file = "dataDump/3dsubGridIndicator_"+str(nn)+".csv"
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
