# cython: profile=True
# cython: linetrace=True

import math
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free

from mpi4py import MPI
comm = MPI.COMM_WORLD
from . import communication
from . import distance
from . import morphology
import sys

cdef int numDirections = 26
cdef int[26][5] directions
directions =  [[-1,-1,-1,  0, 13],
              [-1,-1, 1,  1, 12],
              [-1,-1, 0,  2, 14],
              [-1, 1,-1,  3, 10],
              [-1, 1, 1,  4,  9],
              [-1, 1, 0,  5, 11],
              [-1, 0,-1,  6, 16],
              [-1, 0, 1,  7, 15],
              [-1, 0, 0,  8, 17],
              [ 1,-1,-1,  9,  4],
              [ 1,-1, 1, 10,  3],
              [ 1,-1, 0, 11,  5],
              [ 1, 1,-1, 12,  1],
              [ 1, 1, 1, 13,  0],
              [ 1, 1, 0, 14,  2],
              [ 1, 0,-1, 15,  7],
              [ 1, 0, 1, 16,  6],
              [ 1, 0, 0, 17,  8],
              [ 0,-1,-1, 18, 22],
              [ 0,-1, 1, 19, 21],
              [ 0,-1, 0, 20, 23],
              [ 0, 1,-1, 21, 19],
              [ 0, 1, 1, 22, 18],
              [ 0, 1, 0, 23, 20],
              [ 0, 0,-1, 24, 25],
              [ 0, 0, 1, 25, 24]]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int getBoundaryIDReference(cnp.ndarray[cnp.int8_t, ndim=1] boundaryID):
    cdef int cI,cJ,cK
    cdef int i,j,k
    i = boundaryID[0]
    j = boundaryID[1]
    k = boundaryID[2]

    if i < 0:
        cI = 0
    elif i > 0:
        cI = 9
    else:
        cI = 18

    if j < 0:
        cJ = 0
    elif j > 0:
        cJ = 3
    else:
        cJ = 6

    if k < 0:
        cK = 0
    elif k > 0:
        cK = 1
    else:
        cK = 2

    return cI+cJ+cK


class Set(object):
    def __init__(self, localID = 0, inlet = False, outlet = False, boundary = False, numNodes = 0, numBoundaryNodes = 0):
        self.inlet = inlet
        self.outlet = outlet
        self.boundary = boundary
        self.numNodes = numNodes
        self.localID = localID
        self.globalID = 0
        self.nodes = np.zeros([numNodes,3],dtype=np.int64)
        self.boundaryNodes = np.zeros(numBoundaryNodes,dtype=np.int64)
        self.boundaryFaces = np.zeros(26,dtype=np.uint8)
        self.boundaryNodeID = np.zeros([numBoundaryNodes,3],dtype=np.int64)

    def getNodes(self,n,i,j,k):
        self.nodes[n,0] = i
        self.nodes[n,1] = j
        self.nodes[n,2] = k

    def getBoundaryNodes(self,n,ID,ID2,i,j,k):
        self.boundaryNodes[n] = ID
        self.boundaryFaces[ID2] = 1
        self.boundaryNodeID[n,0] = i
        self.boundaryNodeID[n,1] = j
        self.boundaryNodeID[n,2] = k


class Node(object):
    def __init__(self, ID = 0, localIndex = np.zeros(3,dtype=np.int64), globalIndex = 0, boundary = False, boundaryID = -1, inlet = False, outlet = False ):
        self.ID  = ID
        self.boundary = boundary
        self.inlet  = inlet
        self.outlet = outlet
        self.availDirection = 0
        self.lastDirection = 25
        self.visited = False
        self.localIndex = localIndex
        self.globalIndex = globalIndex
        self.boundaryID = boundaryID
        self.direction = np.zeros(26,dtype='uint8')
        self.nodeDirection = np.zeros(26,dtype='uint64')


class Drainage(object):
    def __init__(self,Domain,Orientation,subDomain,gamma,inlet,edt):
        self.Domain      = Domain
        self.Orientation = Orientation
        self.subDomain   = subDomain
        self.edt         = edt
        self.gamma       = gamma
        self.inletDirection = 0
        self.probeD = 0
        self.probeR = 0
        self.pC = 0
        self.numNWP = 0
        self.ind = None
        self.nwp = None
        self.globalIndexStart = 0
        self.globalBoundarySetID = None
        self.inlet = inlet
        self.matchedSets = []
        self.nwpNodes = 0
        self.totalnwpNodes = np.zeros(1,dtype=np.uint64)
        self.nwpRes = np.zeros([3,2])
        self.Sets = []

    def getDiameter(self,pc):
        if pc == 0:
            self.probeD = 0
            self.probeR = 0
        else:
            self.probeR = 2.*self.gamma/pc
            self.probeD = 2.*self.probeR

    def getpC(self,radius):
        self.pC = 2.*self.gamma/radius

    def probeDistance(self):
        self.ind = np.where( (self.edt.EDT >= self.probeR) & (self.subDomain.grid == 1),1,0).astype(np.uint8)
        self.numNWP = np.sum(self.ind)


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def getNodeInfo(self):
        """
        Speed up Code with avoiding Node Class and Only Use NumPy Arrays
        NodeInfoBin Array is [boundary,inlet,outlet,boundaryID,availDirection,lastDirection,visited]
        NodeInfoIndex Array is [i,j,k,globalIndex]
        NodeDir Array is [directions[26],nodeDirections[26]]
        """

        self.nodeInfoBin = np.zeros([self.numNWP,7],dtype=np.int8)
        self.nodeInfoBin[:,3] = -1 #Initialize BoundaryID
        self.nodeInfoBin[:,5] = 25 #Initialize lastDirection
        cdef cnp.int8_t [:,:] nodeInfoBin
        nodeInfoBin = self.nodeInfoBin

        self.nodeInfoIndex = np.zeros([self.numNWP,7],dtype=np.uint64)
        cdef cnp.uint64_t [:,:] nodeInfoIndex
        nodeInfoIndex = self.nodeInfoIndex

        self.nodeInfoDir = np.zeros([self.numNWP,26],dtype=np.uint8)
        cdef cnp.uint8_t [:,:] nodeInfoDir
        nodeInfoDir = self.nodeInfoDir

        self.nodeInfoDirNode = np.zeros([self.numNWP,26],dtype=np.uint64)
        cdef cnp.uint64_t [:,:] nodeInfoDirNode
        nodeInfoDirNode = self.nodeInfoDirNode

        self.nodeTable = -np.ones_like(self.ind,dtype=np.int64)
        cdef cnp.int64_t [:,:,:] nodeTable
        nodeTable = self.nodeTable

        cdef int c,d,i,j,k,ii,jj,kk,availDirection,perAny,setInlet,setOutlet
        cdef int iLoc,jLoc,kLoc,globIndex
        cdef int iMin,iMax,jMin,jMax,kMin,kMax

        cdef int numFaces,fIndex
        numFaces = self.Orientation.numFaces

        cdef int iStart,jStart,kStart
        iStart = self.subDomain.indexStart[0]
        jStart = self.subDomain.indexStart[1]
        kStart = self.subDomain.indexStart[2]

        cdef int iShape,jShape,kShape
        iShape = self.ind.shape[0]
        jShape = self.ind.shape[1]
        kShape = self.ind.shape[2]

        indP = np.pad(self.ind,1)
        cdef cnp.uint8_t [:,:,:] ind
        ind = indP

        cdef cnp.int64_t [:,:,:] loopInfo
        loopInfo = self.subDomain.loopInfo

        cdef int dN0,dN1,dN2
        dN0 = self.Domain.nodes[0]
        dN1 = self.Domain.nodes[1]
        dN2 = self.Domain.nodes[2]

        c = 0
        for fIndex in range(0,numFaces):
            iMin = loopInfo[fIndex][0][0]
            iMax = loopInfo[fIndex][0][1]
            jMin = loopInfo[fIndex][1][0]
            jMax = loopInfo[fIndex][1][1]
            kMin = loopInfo[fIndex][2][0]
            kMax = loopInfo[fIndex][2][1]
            bID = np.asarray(self.Orientation.faces[fIndex]['ID'],dtype=np.int8)
            perFace  = self.subDomain.neighborPerF[fIndex]
            perAny = perFace.any()
            setInlet = self.subDomain.inlet[fIndex]
            setOutlet = self.subDomain.outlet[fIndex]
            for i in range(iMin,iMax):
                for j in range(jMin,jMax):
                    for k in range(kMin,kMax):
                        if ind[i+1,j+1,k+1] == 1:

                            iLoc = iStart+i
                            jLoc = jStart+j
                            kLoc = kStart+k

                            if iLoc >= dN0:
                                iLoc = 0
                            elif iLoc < 0:
                                iLoc = dN0-1
                            if jLoc >= dN1:
                                jLoc = 0
                            elif jLoc < 0:
                                jLoc = dN1-1
                            if kLoc >= dN2:
                                kLoc = 0
                            elif kLoc < 0:
                                kLoc = dN2-1

                            globIndex = iLoc*dN1*dN2 +  jLoc*dN2 +  kLoc

                            boundaryID = np.copy(bID)
                            if (i < 2):
                                boundaryID[0] = -1
                            elif (i >= iShape-2):
                                boundaryID[0] = 1
                            if (j < 2):
                                boundaryID[1] = -1
                            elif (j >= jShape-2):
                                boundaryID[1] = 1
                            if (k < 2):
                                boundaryID[2] = -1
                            elif(k >= kShape-2):
                                boundaryID[2] = 1

                            boundID = getBoundaryIDReference(boundaryID)
                            nodeInfoBin[c,0] = 1
                            nodeInfoBin[c,1] = setInlet
                            nodeInfoBin[c,2] = setOutlet
                            nodeInfoBin[c,3] = boundID
                            nodeInfoIndex[c,0] = i
                            nodeInfoIndex[c,1] = j
                            nodeInfoIndex[c,2] = k
                            nodeInfoIndex[c,3] = globIndex
                            nodeInfoIndex[c,4] = iLoc
                            nodeInfoIndex[c,5] = jLoc
                            nodeInfoIndex[c,6] = kLoc

                            nodeTable[i,j,k] = c
                            c = c + 1

        iMin = loopInfo[numFaces][0][0]
        iMax = loopInfo[numFaces][0][1]
        jMin = loopInfo[numFaces][1][0]
        jMax = loopInfo[numFaces][1][1]
        kMin = loopInfo[numFaces][2][0]
        kMax = loopInfo[numFaces][2][1]
        for i in range(iMin,iMax):
            for j in range(jMin,jMax):
                for k in range(kMin,kMax):
                    if (ind[i+1,j+1,k+1] == 1):
                        iLoc = iStart+i
                        jLoc = jStart+j
                        kLoc = kStart+k
                        globIndex = iLoc*dN1*dN2 +  jLoc*dN2 +  kLoc
                        nodeInfoIndex[c,0] = i
                        nodeInfoIndex[c,1] = j
                        nodeInfoIndex[c,2] = k
                        nodeInfoIndex[c,3] = globIndex
                        nodeTable[i,j,k] = c
                        c = c + 1

        c = 0
        for fIndex in range(numFaces):
         iMin = loopInfo[fIndex][0][0]
         iMax = loopInfo[fIndex][0][1]
         jMin = loopInfo[fIndex][1][0]
         jMax = loopInfo[fIndex][1][1]
         kMin = loopInfo[fIndex][2][0]
         kMax = loopInfo[fIndex][2][1]
         for i in range(iMin,iMax):
             for j in range(jMin,jMax):
                 for k in range(kMin,kMax):
                     if ind[i+1,j+1,k+1] == 1:
                       availDirection = 0
                       for d in range(0,numDirections):
                           ii = directions[d][0]
                           jj = directions[d][1]
                           kk = directions[d][2]
                           if (ind[i+ii+1,j+jj+1,k+kk+1] == 1):
                               node = nodeTable[i+ii,j+jj,k+kk]
                               nodeInfoDir[c,d] = 1
                               nodeInfoDirNode[c,d] = node
                               availDirection += 1

                       nodeInfoBin[c,4] = availDirection
                       c = c + 1

        iMin = loopInfo[numFaces][0][0]
        iMax = loopInfo[numFaces][0][1]
        jMin = loopInfo[numFaces][1][0]
        jMax = loopInfo[numFaces][1][1]
        kMin = loopInfo[numFaces][2][0]
        kMax = loopInfo[numFaces][2][1]
        for i in range(iMin,iMax):
         for j in range(jMin,jMax):
             for k in range(kMin,kMax):
               if ind[i+1,j+1,k+1] == 1:
                 availDirection = 0
                 for d in range(0,numDirections):
                     ii = directions[d][0]
                     jj = directions[d][1]
                     kk = directions[d][2]
                     if (ind[i+ii+1,j+jj+1,k+kk+1] == 1):
                         node = nodeTable[i+ii,j+jj,k+kk]
                         nodeInfoDir[c,d] = 1
                         nodeInfoDirNode[c,d] = node
                         availDirection += 1

                 nodeInfoBin[c,4] = availDirection
                 c = c + 1

    @cython.boundscheck(False)  # Deactivate bounds checking
    def getConnectedSets(self):
        """
        Connects the NxNxN (or NXN) nodes into connected sets.
        1. Inlet
        2. Outlet
        3. DeadEnd
        """
        cdef int node,ID,nodeValue,d,oppDir,avail,n,index,bN

        cdef int numNWP,numSetNodes,numNodes,numBoundNodes,setCount
        numNWP = self.numNWP
        numSetNodes = 0
        numNodes = 0
        numBoundNodes = 0
        setCount = 0

        setInlet = False
        setBoundary = False

        _nodeIndex = np.zeros([self.numNWP,8],dtype=np.int64)
        cdef cnp.int64_t [:,::1] nodeIndex
        nodeIndex = _nodeIndex

        for i in range(numNWP):
          nodeIndex[i,3] = -1

        cdef cnp.int8_t [:,:] nodeInfoBin
        nodeInfoBin = self.nodeInfoBin

        cdef cnp.uint64_t [:,:] nodeInfoIndex
        nodeInfoIndex = self.nodeInfoIndex

        cdef cnp.uint8_t [:,:] nodeInfoDir
        nodeInfoDir = self.nodeInfoDir

        cdef cnp.uint64_t [:,:] nodeInfoDirNode
        nodeInfoDirNode = self.nodeInfoDirNode

### DELETE THIS LINE?
        cdef cnp.int64_t [:,:,:] nodeTable
        nodeTable = self.nodeTable

        cdef cnp.int8_t [:] currentNode

        ### Loop Through All Nodes
        for node in range(0,numNWP):
            if nodeInfoBin[node,6] == 1:
                pass
            else:
                queue=[node]
                while queue:
                    ID = queue.pop(-1)
                    currentNode = nodeInfoBin[ID]
                    if currentNode[6]:
                        pass
                    else:
                        nodeIndex[numNodes,0] = nodeInfoIndex[ID,0]
                        nodeIndex[numNodes,1] = nodeInfoIndex[ID,1]
                        nodeIndex[numNodes,2] = nodeInfoIndex[ID,2]

                        nodeIndex[numNodes,5] = nodeInfoIndex[ID,4]
                        nodeIndex[numNodes,6] = nodeInfoIndex[ID,5]
                        nodeIndex[numNodes,7] = nodeInfoIndex[ID,6]


                        if currentNode[0]:
                            setBoundary = True
                            numBoundNodes = numBoundNodes + 1
                            nodeIndex[numNodes,3] = currentNode[3]
                            nodeIndex[numNodes,4] = nodeInfoIndex[ID,3]
                            if currentNode[1]:
                                setInlet = True
                        numSetNodes = numSetNodes + 1
                        numNodes = numNodes + 1
                        while (currentNode[4] > 0):

                            nodeValue = -1
                            if currentNode[4] > 0:
                                d = currentNode[5]
                                found = 0
                                while d >= 0  and not found:
                                  if nodeInfoDir[ID,d] == 1:
                                    found = 1
                                    oppDir = directions[d][4]
                                    nodeValue = nodeInfoDirNode[ID,d]
                                    nodeInfoDir[nodeValue,oppDir] = 0
                                    currentNode[4] = currentNode[4] - 1
                                    currentNode[5] = d
                                    nodeInfoDir[ID,d] = 0
                                  else:
                                    d = d - 1
                            if (nodeValue > -1):
                                if nodeInfoBin[nodeValue,6] or nodeInfoBin[nodeValue,4] == 0:
                                    pass
                                else:
                                    queue.append(nodeValue)
                            else:
                                break
                        currentNode[6] = 1

                if setCount == 0:
                    self.Sets = [Set(localID = setCount,
                                   inlet = setInlet,
                                   outlet = 0,
                                   boundary = setBoundary,
                                   numNodes = numSetNodes,
                                   numBoundaryNodes = numBoundNodes)]
                else:
                    self.Sets.append(Set(localID = setCount,
                                       inlet = setInlet,
                                       outlet = 0,
                                       boundary = setBoundary,
                                       numNodes = numSetNodes,
                                       numBoundaryNodes = numBoundNodes))

                bN = 0
                for n in range(0,self.Sets[setCount].numNodes):
                    index = numNodes-numSetNodes+n
                    self.Sets[setCount].getNodes(n,nodeIndex[index,0],nodeIndex[index,1],nodeIndex[index,2])
                    if nodeIndex[index,3] > -1:
                        self.Sets[setCount].getBoundaryNodes(bN,nodeIndex[index,4],nodeIndex[index,3],nodeIndex[index,5],nodeIndex[index,6],nodeIndex[index,7])
                        bN = bN + 1

                setCount = setCount + 1
                numSetNodes = 0
                numBoundNodes = 0
                setInlet = False
                setBoundary = False

        self.setCount = setCount

    def getBoundarySets(self):
        """
        Only Get the Sets the are on a valid subDomain Boundary.
        Organize data so sending procID, boundary nodes.
        """

        nI = self.subDomain.subID[0] + 1  # PLUS 1 because lookUpID is Padded
        nJ = self.subDomain.subID[1] + 1  # PLUS 1 because lookUpID is Padded
        nK = self.subDomain.subID[2] + 1  # PLUS 1 because lookUpID is Padded

        self.boundaryData = {self.subDomain.ID: {'NeighborProcID':{}}}

        bSetCount = 0
        self.boundarySets = [Set()]

        for numSets in range(0,self.setCount):
            if self.Sets[numSets].boundary:
                self.boundarySets[bSetCount] = self.Sets[numSets]
                bSetCount = bSetCount + 1
                self.boundarySets.append(Set())

        self.boundarySets.pop()

        for bSet in self.boundarySets[:]:
            for face in range(0,numDirections):
                if bSet.boundaryFaces[face] > 0:

                    i = directions[face][0]
                    j = directions[face][1]
                    k = directions[face][2]

                    neighborProc = self.subDomain.lookUpID[i+nI,j+nJ,k+nK]
                    if neighborProc == -1:
                        bSet.boundaryFaces[face] = 0
                    else:
                        if neighborProc not in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                            self.boundaryData[self.subDomain.ID]['NeighborProcID'][neighborProc] = {'setID':{}}
                        self.boundaryData[self.subDomain.ID]['NeighborProcID'][neighborProc]['setID'][bSet.localID] = {'boundaryNodes':bSet.boundaryNodes,'ProcID':self.subDomain.ID,'inlet':bSet.inlet}

            if (np.sum(bSet.boundaryFaces) == 0):
                self.boundarySets.remove(bSet)
                bSet.boundary = False

        self.boundSetCount = len(self.boundarySets)

    def correctBoundarySets(self):

        otherBD = {}


        ### Sort Out Own Proc Bondary Data and Other Procs Boundary Data
        countOwnSets = 0
        countOtherSets = 0
        for procID in self.boundaryData.keys():
            if procID == self.subDomain.ID:
                ownBD = self.boundaryData[procID]
                for nbProc in ownBD['NeighborProcID'].keys():
                    for ownSet in ownBD['NeighborProcID'][nbProc]['setID'].keys():
                        countOwnSets = countOwnSets + 1
            else:
                otherBD[procID] = self.boundaryData[procID]
                for otherSet in otherBD[procID]['NeighborProcID'][procID]['setID'].keys():
                        countOtherSets = countOtherSets + 1

        numSets = np.max([countOwnSets,countOtherSets])

        ### Loop through own Proc Boundary Data to Find a Match
        c = 0
        for nbProc in ownBD['NeighborProcID'].keys():
            for ownSet in ownBD['NeighborProcID'][nbProc]['setID'].keys():
                ownNodes = ownBD['NeighborProcID'][nbProc]['setID'][ownSet]['boundaryNodes']
                ownInlet = ownBD['NeighborProcID'][nbProc]['setID'][ownSet]['inlet']
                otherSetKeys = list(otherBD[nbProc]['NeighborProcID'][nbProc]['setID'].keys())
                numOtherSetKeys = len(otherSetKeys)

                testSetKey = 0
                matchedOut = False
                while testSetKey < numOtherSetKeys:
                    inlet = False
                    otherNodes = otherBD[nbProc]['NeighborProcID'][nbProc]['setID'][otherSetKeys[testSetKey]]['boundaryNodes']
                    otherInlet = otherBD[nbProc]['NeighborProcID'][nbProc]['setID'][otherSetKeys[testSetKey]]['inlet']

                    if len(set(ownNodes).intersection(otherNodes)) > 0:
                        if (ownInlet or otherInlet):
                            inlet = True
                        self.matchedSets.append([self.subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet])
                        matchedOut = True
                    testSetKey = testSetKey + 1

                if not matchedOut:
                    print("Set Not Matched! Hmmm",self.subDomain.ID,nbProc,ownSet,ownNodes)

    def organizeSets(self,size,drainData):
        """
        Input: [drain.matchedSets,drain.setCount,drain.boundSetCount] from all Procs
        Matched Sets contains [subDomain.ID,ownSetID,neighProc,neighSetID,Inlet]

        Output: globalIndexStart,globalBoundarySetID
        """

        if self.subDomain.ID == 0:

            #Gather all information from all Procs
            allMatchedSets = np.zeros([0,5],dtype=np.int64)
            numSets = np.zeros(size,dtype=np.int64)
            numBoundSets = np.zeros(size,dtype=np.int64)
            for n in range(0,size):
                numSets[n] = drainData[n][1]
                numBoundSets[n] = drainData[n][2]
                if numBoundSets[n] > 0:
                    allMatchedSets = np.append(allMatchedSets,drainData[n][0],axis=0)

            ### Propagate Inlet Info
            for s in allMatchedSets:
                if s[4] == 1:
                    indexs = np.where( (allMatchedSets[:,0]==s[2])
                                     & (allMatchedSets[:,1]==s[3]))[0].tolist()
                    while indexs:
                        ind = indexs.pop()
                        addIndexs  = np.where( (allMatchedSets[:,0]==allMatchedSets[ind,2])
                                             & (allMatchedSets[:,1]==allMatchedSets[ind,3])
                                             & (allMatchedSets[:,4]==0) )[0].tolist()
                        if addIndexs:
                            indexs.extend(addIndexs)
                        allMatchedSets[ind,4] = 1

            ### Get Unique Entries and Inlet T(1)/F(0)
            globalSetList = []
            globalInletList = []
            for s in allMatchedSets:
                if [s[0],s[1]] not in globalSetList:
                    globalSetList.append([s[0],s[1]])
                    globalInletList.append(s[4])
                else:
                    ind = globalSetList.index([s[0],s[1]])
                    globalInletList[ind] = s[4]
            globalSetID = np.c_[np.asarray(globalSetList),-np.ones(len(globalSetList)),np.asarray(globalInletList)]

            cID = 0
            for s in allMatchedSets:
                ind = np.where( (globalSetID[:,0]==s[0]) & (globalSetID[:,1]==s[1]))
                if (globalSetID[ind,2] < 0):
                    indNeigh = np.where( (globalSetID[:,0]==s[2]) & (globalSetID[:,1]==s[3]))
                    if (globalSetID[indNeigh,2] < 0):
                        globalSetID[indNeigh,2] = cID
                        globalSetID[ind,2] = cID
                        cID = cID + 1
                    elif (globalSetID[indNeigh,2] > -1):
                        globalSetID[ind,2] = globalSetID[indNeigh,2]
                elif (globalSetID[ind,2] > -1):
                    indNeigh = np.where( (globalSetID[:,0]==s[2]) & (globalSetID[:,1]==s[3]))
                    if (globalSetID[indNeigh,2] < 0):
                        globalSetID[indNeigh,2] = globalSetID[ind,2]

            localSetStart = np.zeros(size,dtype=np.int64)
            globalSetScatter = [globalSetID[np.where(globalSetID[:,0]==0)]]
            localSetStart[0] = cID
            for n in range(1,size):
                localSetStart[n] = localSetStart[n-1] + numSets[n-1] - numBoundSets[n-1]
                globalSetScatter.append(globalSetID[np.where(globalSetID[:,0]==n)])
        else:
            localSetStart = None
            globalSetScatter = None

        self.globalIndexStart = comm.scatter(localSetStart, root=0)
        self.globalBoundarySetID = comm.scatter(globalSetScatter, root=0)

    def updateSetID(self):
        """
        globalBoundarySetID = [subDomain.ID,setLocalID,globalID,Inlet]
        """
        c = 0
        for s in self.Sets:
            if s.boundary == True:
                ind = np.where(self.globalBoundarySetID[:,1]==s.localID)[0][0]
                s.globalID = self.globalBoundarySetID[ind,2]
                s.inlet = self.globalBoundarySetID[ind,3]
            else:
                s.globalID = self.globalIndexStart + c
                c = c + 1

    def getNWP(self):
        NWNodes = []
        self.nwp = np.zeros_like(self.ind)

        for s in self.Sets:
            if s.inlet:
                for node in s.nodes:
                    NWNodes.append(node)

        for n in NWNodes:
            self.nwp[n[0],n[1],n[2]] = 1

    def finalizeNWP(self,nwpDist):

        if self.nwpRes[0,0]:
            nwpDist = nwpDist[-1:,:,:]
        elif self.nwpRes[0,1]:
            nwpDist = nwpDist[1:,:,:]
        self.nwpFinal = np.copy(self.subDomain.grid)
        self.nwpFinal = np.where( (nwpDist ==  1) & (self.subDomain.grid == 1),2,self.nwpFinal)
        self.nwpFinal = np.where( (self.subDomain.res == 1),2,self.nwpFinal)

        own = self.subDomain.ownNodes
        ownGrid =  self.nwpFinal[own[0][0]:own[0][1],
                             own[1][0]:own[1][1],
                             own[2][0]:own[2][1]]
        self.nwpNodes = np.sum(np.where(ownGrid==2,1,0))

    def drainCOMM(self):
        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = communication.subDomainComm(self.Orientation,self.subDomain,self.boundaryData[self.subDomain.ID]['NeighborProcID'])

    def drainCOMMUnpack(self):

        #### Faces ####
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.boundaryData:
                        self.boundaryData[neigh] = {'NeighborProcID':{}}
                    self.boundaryData[neigh]['NeighborProcID'][neigh] = self.dataRecvFace[fIndex]

        #### Edges ####
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.boundaryData:
                        self.boundaryData[neigh] = {'NeighborProcID':{}}
                    self.boundaryData[neigh]['NeighborProcID'][neigh] = self.dataRecvEdge[eIndex]


        #### Corners ####
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.boundaryData:
                        self.boundaryData[neigh] = {'NeighborProcID':{}}
                    self.boundaryData[neigh]['NeighborProcID'][neigh] = self.dataRecvCorner[cIndex]



def calcDrainage(rank,size,pc,domain,subDomain,inlet,EDT,info = False):

    for p in pc:
        if p == 0:
            sW = 1
        else:
            drain = Drainage(Domain = domain, Orientation = subDomain.Orientation, subDomain = subDomain, edt = EDT, gamma = 1., inlet = inlet)
            if info:
                drain.getpC(EDT.maxD)
                print("Minimum pc",drain.pC)
                pCMax = drain.getpC(EDT.minD)
                print("Maximum pc",drain.pC)

            drain.getDiameter(p)
            drain.probeDistance()
            numNWPSum = np.zeros(1,dtype=np.uint64)
            comm.Allreduce( [drain.numNWP, MPI.INT], [numNWPSum, MPI.INT], op = MPI.SUM )
            if numNWPSum < 1:
                drain.nwp = np.copy(subDomain.grid)
                drain.nwpFinal = drain.nwp
            else:
                drain.getNodeInfo()
                drain.getConnectedSets()
                if size > 1:
                    drain.getBoundarySets()
                    drain.drainCOMM()
                    drain.drainCOMMUnpack()
                    drain.correctBoundarySets()
                    drainData = [drain.matchedSets,drain.setCount,drain.boundSetCount]
                    drainData = comm.gather(drainData, root=0)
                    drain.organizeSets(size,drainData)
                    drain.updateSetID()
                drain.getNWP()
                morphL = morphology.morph(rank,size,domain,subDomain,drain.nwp,drain.probeR)
                drain.finalizeNWP(morphL.gridOut)

                numNWPSum = np.zeros(1,dtype=np.uint64)
                comm.Allreduce( [drain.nwpNodes, MPI.INT], [drain.totalnwpNodes, MPI.INT], op = MPI.SUM )
        if rank == 0:
            sW = 1.-drain.totalnwpNodes[0]/subDomain.totalPoreNodes[0]
            print("Wetting phase saturation is: %e at pC of %e" %(sW,p))

    return drain,morphL
