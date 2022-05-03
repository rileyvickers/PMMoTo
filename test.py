import numpy as np
from mpi4py import MPI
from scipy import ndimage
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

from MAtools import printMedialAxis
from scipy.spatial import KDTree
import pdb
import edt
import sys
import time

import PMMoTo


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    start_time = time.time()

subDomains = [2,2,2]
nodes = [51,51,51]
periodic = [False,False,False]
domainFile = open('testDomains/pack_sub.out', 'r')

numSubDomains = np.prod(subDomains)

domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,periodic,domainFile,PMMoTo.readPorousMediaXYZR)
sDEDTL = PMMoTo.calcEDT(rank,domain,subDomains,sDL)

if rank == 0:
    print("--- %s seconds ---" % (time.time() - start_time))
    sD = np.empty((numSubDomains), dtype = object)
    sDEDT = np.empty((numSubDomains), dtype = object)
    sD[0] = sDL
    sDEDT[0] = sDEDTL
    for neigh in range(1,numSubDomains):
        sD[neigh] = comm.recv(source=neigh)
        sDEDT[neigh] = comm.recv(source=neigh)

if rank > 0:
    for neigh in range(1,numSubDomains):
        if rank == neigh:
            comm.send(sDL,dest=0)
            comm.send(sDEDTL,dest=0)



if rank==0:
    testAlgo = True
    if testAlgo:

        startTime = time.time()

        domainFile.close()
        domainFile = open('testDomains/pack_sub.out', 'r')
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

        if periodic[0] and not periodic[1] and not periodic[2]:
            gridOut = gridOut[pG[0]:-pG[0],:,:]
            realDT = realDT[pG[0]:-pG[0],:,:]
        elif not periodic[0] and periodic[1] and not periodic[2]:
            gridOut = gridOut[:,pG[1]:-pG[1],:]
            realDT = realDT[:,pG[1]:-pG[1],:]
        elif not periodic[0] and not periodic[1] and periodic[2]:
            gridOut = gridOut[:,:,pG[2]:-pG[2]]
            realDT = realDT[:,:,pG[2]:-pG[2]]
        elif periodic[0] and not periodic[1] and periodic[2]:
            gridOut = gridOut[pG[0]:-pG[0],:,pG[2]:-pG[2]]
            realDT = realDT[pG[0]:-pG[0],:,pG[2]:-pG[2]]
        elif periodic[0] and periodic[1] and not periodic[2]:
            gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],:]
            realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],:]
        elif not periodic[0] and periodic[1] and periodic[2]:
            gridOut = gridOut[:,pG[1]:-pG[1],pG[2]:-pG[2]]
            realDT = realDT[:,pG[1]:-pG[1],pG[2]:-pG[2]]
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
        print("L2 EDT Error Norm",np.linalg.norm(diffEDT) )

    # printGridOut = np.zeros([checkEDT.size,5])
    # c = 0
    # for i in range(0,checkEDT.shape[0]):
    #     for j in range(0,checkEDT.shape[1]):
    #         for k in range(0,checkEDT.shape[2]):
    #             if  checkGrid[i,j,k] == 1:
    #                 printGridOut[c,0] = x[i]#sDAll[nn].x[i]#sDAll[nn].indexStart[0] + i #sDAll[nn].x[i]
    #                 printGridOut[c,1] = y[j]#sDAll[nn].y[j]#sDAll[nn].indexStart[1] + j#sDAll[nn].y[j]
    #                 printGridOut[c,2] = z[k]#sDAll[nn].z[k]#sDAll[nn].indexStart[2] + k#sDAll[nn].z[k]
    #                 printGridOut[c,3] = diffEDT[i,j,k]
    #                 printGridOut[c,4] = checkEDT[i,j,k]
    #                 c = c + 1
    #
    # header = "x,y,z,DiffEDT,EDT"
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
    # orient = Orientation()
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
