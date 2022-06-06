# cython: profile=True

import math
import numpy as np
cimport numpy as cnp
cimport cython

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
cdef getDirection3D(int ID,
                    int i,
                    int j,
                    int k,
                    int availDirection,
                    cnp.ndarray[cnp.uint8_t, ndim=1] direction,
                    cnp.ndarray[cnp.uint64_t, ndim=1] nodeDirection):

  cdef int d,ii,jj,kk,oppDir,returnCell,found
  cdef int returnVal[4]

  if availDirection > 0:
      d = 25
      found = 0
      while d > 0  and not found:
        if direction[d] == 1:
          found = 1
        else:
          d = d - 1
      ii = directions[d][0]
      jj = directions[d][1]
      kk = directions[d][2]
      oppDir = directions[d][4]
      returnCell = nodeDirection[d]
      availDirection = availDirection - 1
  else:
      returnCell = -1


  return returnCell,d,oppDir,availDirection

class Set(object):
    def __init__(self, localID = 0):
        self.inlet = False
        self.outlet = False
        self.boundary = False
        self.nodes = []
        self.numNodes = 0
        self.boundaryFaces = []
        self.boundaryNodes = []
        self.globalID = 0
        self.localID = localID
        self.boundaryNodesGlobalIndex = []

    def getUniqueBoundaryFace(self,subDomain):
        """
        Account for when Boundary Node is on subDomain that is on the boundary of the Domain. If boundaryNode
        is on edge or corner, we still need to transfer infromation to faces.
        """
        perm = [[0,1],[1,2],[0,2]]
        for n in self.boundaryNodes:
            if n.boundaryID not in self.boundaryFaces:
                self.boundaryFaces.append(n.boundaryID)

            bCount = np.sum(np.abs(n.boundaryID))

            nI = subDomain.subID[0] + 1  # PLUS 1 because lookUpID is Padded
            nJ = subDomain.subID[1] + 1  # PLUS 1 because lookUpID is Padded
            nK = subDomain.subID[2] + 1  # PLUS 1 because lookUpID is Padded
            i = n.boundaryID[0]
            j = n.boundaryID[1]
            k = n.boundaryID[2]
            neighborProc = subDomain.lookUpID[i+nI,j+nJ,k+nK]

            if (neighborProc < 0 and bCount > 1):
                if bCount == 3:
                    for i in range(0,3):
                        face = n.boundaryID[:]
                        face[i] = 0
                        if face not in self.boundaryFaces:
                            self.boundaryFaces.append(face)
                for p in perm:
                    face = n.boundaryID[:]
                    face[p[0]] = 0
                    face[p[1]] = 0
                    if (np.sum(np.abs(face)) != 0 and face not in self.boundaryFaces):
                        self.boundaryFaces.append(face)

class Node(object):
    def __init__(self, ID = None, coords = None, localIndex = None, globalIndex = None, dist = None, boundary = False, boundaryID = None, inlet = False, outlet = False ):
        self.ID  = ID
        self.coords = coords
        self.localIndex = localIndex
        self.globalIndex = globalIndex
        self.boundary = boundary
        self.boundaryID = boundaryID
        self.inlet  = inlet
        self.outlet = outlet
        self.dist   = dist
        self.direction = np.zeros(26,dtype='uint8')
        self.nodeDirection = np.zeros(26,dtype='uint64')
        self.availDirection = 0
        self.visited = False
        self.medialNode = False
        self.endPath = False

    def get_id(self):
        return self.id

    def validDirection(self,c):
        self.direction[c] = 1

    def setNodeDirection(self,c,node):
        self.nodeDirection[c] = node

    def saveDirection(self):
        self.saveDirection = np.copy(self.direction)

    def getAvailDirections(self):
        self.availDirection = int(self.direction.sum())

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
        self.ind = None
        self.nwp = None
        self.globalIndexStart = 0
        self.globalBoundarySetID = None
        self.inlet = inlet
        self.matchedSets = []
        self.nwpNodes = 0
        self.totalnwpNodes = np.zeros(1,dtype=np.uint64)
        self.boundInlet = 0
        self.nwpRes = np.zeros([3,2])

    def getDiameter(self,pc):
        if pc == 0:
            self.probeD = 0
            self.probeR = 0
        else:
            self.probeR = 2.*self.gamma/pc
            self.probeD = 2.*self.probeR

    def probeDistance(self):
        self.ind = np.where( (self.edt.EDT >= self.probeR) & (self.subDomain.grid == 1),1,0).astype(np.uint8)
        self.numNWP = np.sum(self.ind)

    def getNodeInfo(self):

        self.nodeInfo = np.empty((self.numNWP), dtype = object)

        self.nodeTable = -np.ones_like(self.ind,dtype=np.int64)
        cdef cnp.int64_t [:,:,:] nodeTable
        nodeTable = self.nodeTable

        cdef int i,j,k,c,d,ii,jj,kk,availDirection
        cdef int iLoc,jLoc,kLoc,globIndex

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

        cdef cnp.uint8_t [:,:,:] ind
        ind = self.ind

        cdef cnp.int64_t [:,:,:] loopInfo
        loopInfo = self.subDomain.loopInfo

        cdef int iMin,iMax
        cdef int jMin,jMax
        cdef int kMin,kMax

        c = 0
        for fIndex in range(0,numFaces):
            iMin = loopInfo[fIndex][0][0]
            iMax = loopInfo[fIndex][0][1]
            jMin = loopInfo[fIndex][1][0]
            jMax = loopInfo[fIndex][1][1]
            kMin = loopInfo[fIndex][2][0]
            kMax = loopInfo[fIndex][2][1]
            bID = list(self.Orientation.faces[fIndex]['ID'])
            for i in range(iMin,iMax):
                for j in range(jMin,jMax):
                    for k in range(kMin,kMax):
                        if ind[i,j,k] == 1:

                            iLoc = iStart+i
                            jLoc = jStart+j
                            kLoc = kStart+k

                            perFace  = self.subDomain.neighborPerF[fIndex]

                            if perFace.any():
                                if iLoc >= self.Domain.nodes[0]:
                                    iLoc = 0
                                elif iLoc < 0:
                                    iLoc = self.Domain.nodes[0]-1
                                if jLoc >= self.Domain.nodes[1]:
                                    jLoc = 0
                                elif jLoc < 0:
                                    jLoc = self.Domain.nodes[1]-1
                                if kLoc >= self.Domain.nodes[2]:
                                    kLoc = 0
                                elif kLoc < 0:
                                    kLoc = self.Domain.nodes[2]-1

                            globIndex = iLoc*self.Domain.nodes[1]*self.Domain.nodes[2] +  jLoc*self.Domain.nodes[2] +  kLoc

                            boundaryID = bID.copy()
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

                            self.nodeInfo[c] = Node(ID=c,
                                                    localIndex = np.array([i,j,k],dtype=np.uint64),
                                                    globalIndex = globIndex,
                                                    boundary = True,
                                                    boundaryID = boundaryID,
                                                    inlet = self.subDomain.inlet[fIndex],
                                                    outlet = self.subDomain.outlet[fIndex])
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
                    if (ind[i,j,k] == 1):
                        iLoc = iStart+i
                        jLoc = jStart+j
                        kLoc = kStart+k
                        globIndex = iLoc*self.Domain.nodes[1]*self.Domain.nodes[2] +  jLoc*self.Domain.nodes[2] +  kLoc
                        self.nodeInfo[c] = Node(ID = c,
                                                localIndex =  np.array([0,0,0],dtype=np.uint64),
                                                globalIndex = globIndex,
                                                boundary = False,
                                                boundaryID = [0,0,0],
                                                inlet = False,
                                                outlet = False)
                        nodeTable[i,j,k] = c
                        c = c + 1

    def getNodeDirections(self):

     indP = np.pad(self.ind,1)
     cdef cnp.uint8_t [:,:,:] ind
     ind = indP

     cdef cnp.int64_t [:,:,:] nodeTable
     nodeTable = self.nodeTable

     cdef cnp.int64_t [:,:,:] loopInfo
     loopInfo = self.subDomain.loopInfo

     cdef int i,j,k,c,d,ii,jj,kk,availDirection,node

     cdef int numFaces,fIndex
     numFaces = self.Orientation.numFaces

     cdef int iMin,iMax
     cdef int jMin,jMax
     cdef int kMin,kMax

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
                           ii = directions[d][0] + 1
                           jj = directions[d][1] + 1
                           kk = directions[d][2] + 1
                           if (ind[i+ii,j+jj,k+kk] == 1):
                               self.nodeInfo[c].direction[d] = 1
                               node = nodeTable[i+ii-1,j+jj-1,k+kk-1]
                               self.nodeInfo[c].nodeDirection[d] = node
                               availDirection += 1

                       self.nodeInfo[c].availDirection = availDirection
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
                     ii = directions[d][0] + 1
                     jj = directions[d][1] + 1
                     kk = directions[d][2] + 1
                     if (ind[i+ii,j+jj,k+kk] == 1):
                         self.nodeInfo[c].direction[d] = 1
                         node = nodeTable[i+ii-1,j+jj-1,k+kk-1]
                         self.nodeInfo[c].nodeDirection[d] = node
                         availDirection += 1

                 self.nodeInfo[c].availDirection = availDirection
                 c = c + 1

    def getConnectedSets(self):
        """
        Connects the NxNxN (or NXN) nodes into connected sets.
        1. Inlet
        2. Outlet
        3. DeadEnd
        """
        cdef int node,ID,nodeValue,d,oppDir,avail,numNWP,numNodes,setCount
        cdef list queue
        cdef object currentNode
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] localIndex

        numNWP = self.numNWP
        self.Sets = [Set(localID = 0)]
        numNodes = 0
        setCount = 0


        ### Loop Through All Nodes
        for node in range(0,numNWP):
            queue=[node]

            if self.nodeInfo[queue[-1]].visited:
                queue.pop(-1)
            else:
                while queue:
                    ID = queue.pop(-1)
                    currentNode = self.nodeInfo[ID]

                    if currentNode.visited:
                        pass
                    else:

                        self.Sets[setCount].nodes.append(currentNode)
                        numNodes = numNodes + 1

                        if currentNode.boundary:
                            self.Sets[setCount].boundary = True
                            self.Sets[setCount].boundaryNodes.append(currentNode)
                            self.Sets[setCount].boundaryNodesGlobalIndex.append(currentNode.globalIndex)

                            if currentNode.inlet:
                                self.Sets[setCount].inlet = True

                        avail = currentNode.availDirection

                        while (avail > 0):
                            nodeValue,d,oppDir,avail = getDirection3D(ID,
                                                                      currentNode.localIndex[0],
                                                                      currentNode.localIndex[1],
                                                                      currentNode.localIndex[2],
                                                                      currentNode.availDirection,
                                                                      currentNode.direction,
                                                                      currentNode.nodeDirection)

                            self.nodeInfo[nodeValue].direction[oppDir] = 0
                            currentNode.direction[d] = 0
                            currentNode.availDirection = avail

                            if (nodeValue > -1):
                                if self.nodeInfo[nodeValue].visited:
                                    pass
                                else:
                                    queue.append(nodeValue)
                            else:
                                break

                        currentNode.visited = True

                self.Sets[setCount].numNodes = numNodes-1
                setCount = setCount + 1
                self.Sets.append(Set(localID = setCount))
                numNodes = 0

        self.setCount = setCount
        self.Sets.pop()

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
                self.boundarySets[bSetCount].getUniqueBoundaryFace(self.subDomain)
                bSetCount = bSetCount + 1
                self.boundarySets.append(Set())


        self.boundarySets.pop()

        for bSet in self.boundarySets[:]:
            for face in bSet.boundaryFaces[:]:
                i = face[0]
                j = face[1]
                k = face[2]

                neighborProc = self.subDomain.lookUpID[i+nI,j+nJ,k+nK]
                if neighborProc == -1:
                    bSet.boundaryFaces.remove(face)
                else:
                    if neighborProc not in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                        self.boundaryData[self.subDomain.ID]['NeighborProcID'][neighborProc] = {'setID':{}}
                    if bSet.inlet:
                        self.boundInlet = 1
                    self.boundaryData[self.subDomain.ID]['NeighborProcID'][neighborProc]['setID'][bSet.localID] = {'boundaryNodes':bSet.boundaryNodesGlobalIndex,'ProcID':self.subDomain.ID,'inlet':bSet.inlet}

            if (len(bSet.boundaryFaces) == 0):
                self.boundarySets.remove(bSet)
                bSet.boundary = False

        self.boundSetCount = len(self.boundarySets)

    def correctBoundarySets(self):

        otherBD = {}

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

        c = 0
        for nbProc in ownBD['NeighborProcID'].keys():
            for ownSet in ownBD['NeighborProcID'][nbProc]['setID'].keys():
                ownNodes = ownBD['NeighborProcID'][nbProc]['setID'][ownSet]['boundaryNodes']
                ownInlet = ownBD['NeighborProcID'][nbProc]['setID'][ownSet]['inlet']
                numOwnNodes = len(ownNodes)
                otherSetKeys = list(otherBD[nbProc]['NeighborProcID'][nbProc]['setID'].keys())
                numOtherSetKeys = len(otherSetKeys)

                inlet = 0
                testSetKey = 0
                matchedOut = False
                while testSetKey < numOtherSetKeys:
                    otherNodes = otherBD[nbProc]['NeighborProcID'][nbProc]['setID'][otherSetKeys[testSetKey]]['boundaryNodes']
                    otherInlet = otherBD[nbProc]['NeighborProcID'][nbProc]['setID'][otherSetKeys[testSetKey]]['inlet']

                    if len(set(ownNodes).intersection(otherNodes)) > 0:
                        if (ownInlet or otherInlet):
                            inlet = 1
                        self.matchedSets.append([self.subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet])
                        matchedOut = True
                    testSetKey = testSetKey + 1

                if not matchedOut:
                    print("Set Not Matched! Hmmm",self.subDomain.ID,nbProc,ownSet,ownNodes)

    def organizeSets(self,size,drainData):

        if self.subDomain.ID == 0:
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
                    NWNodes.append(node.localIndex)

        for n in NWNodes:
            self.nwp[n[0],n[1],n[2]] = 1

        # if (self.subDomain.boundaryID[0] == -1 and self.Domain.inlet[0] == -1):
        #     #self.nwp = np.pad(self.nwp,( (0,1), (0,0), (0,0) ), 'constant', constant_values=1)
        #     #self.nwpRes[0,0] = 1
        #     self.nwp[0,:,:] = 1
        # if (self.subDomain.boundaryID[0] == 1 and self.Domain.inlet[0] == 1):
        #     pass#self.nwp = np.pad(self.nwp,( (1,0), (0,0), (0,0) ), 'constant', constant_values=1)
        #     #self.nwpRes[0,1] = 1
        #     #pass#self.nwp[-1,:,:] = 1
        # if (self.subDomain.boundaryID[1] == -1 and self.Domain.inlet[1] == -1):
        #     self.nwp[:,0,:] = 1
        # if (self.subDomain.boundaryID[1] == 1 and self.Domain.inlet[1] == 1):
        #     self.nwp[:,-1,:] = 1
        # if (self.subDomain.boundaryID[2] == -1 and self.Domain.inlet[2] == -1):
        #     self.nwp[:,:,0] = 1
        # if (self.subDomain.boundaryID[2] == 1 and self.Domain.inlet[2] == 1):
        #     self.nwp[:,:,-1] = 1

    def finalizeNWP(self,nwpDist):

        if self.nwpRes[0,0]:
            nwpDist = nwpDist[-1:,:,:]
        elif self.nwpRes[0,1]:
            nwpDist = nwpDist[1:,:,:]
        self.nwpFinal = np.where( (nwpDist ==  1) & (self.subDomain.grid == 1),1,0)
        own = self.subDomain.ownNodes
        ownGrid =  self.nwpFinal[own[0][0]:own[0][1],
                             own[1][0]:own[1][1],
                             own[2][0]:own[2][1]]
        self.nwpNodes = np.sum(ownGrid)

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



def calcDrainage(rank,size,pc,domain,subDomain,inlet,EDT):

    for p in pc:
        if p == 0:
            sW = 1
        else:
            drain = Drainage(Domain = domain, Orientation = subDomain.Orientation, subDomain = subDomain, edt = EDT, gamma = 1., inlet = inlet)
            drain.getDiameter(p)
            drain.probeDistance()
            numNWPSum = np.zeros(1,dtype=np.uint64)
            comm.Allreduce( [drain.numNWP, MPI.INT], [numNWPSum, MPI.INT], op = MPI.SUM )

            if numNWPSum < 1:
                drain.nwp = np.copy(subDomain.grid)
                drain.nwpFinal = drain.nwp
            else:
                drain.getNodeInfo()
                drain.getNodeDirections()
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
            #print("Wetting phase saturation is: %e at pC of %e" %(sW,p))
            print(p,sW,drain.totalnwpNodes[0],subDomain.totalPoreNodes[0],drain.probeR)

    return drain
