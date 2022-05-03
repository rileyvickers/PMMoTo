import numpy as np
from mpi4py import MPI
from scipy.spatial import KDTree
import edt
import pdb
comm = MPI.COMM_WORLD


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
        self.availDirection = 0
        self.visited = False
        self.medialNode = False
        self.endPath = False

    def get_id(self):
        return self.id

    def validDirection(self,c):
        self.direction[c] = 1

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
        self.ind = None
        self.nwp = None
        self.globalIndexStart = 0
        self.globalBoundarySetID = None
        self.inlet = inlet
        self.matchedSets = []

    def getDiameter(self,pc):
        if pc == 0:
            self.probeD = 0
        else:
            self.probeD = 4*self.gamma/pc

    def probeDistance(self):
        self.ind = np.where(self.edt.EDT > self.probeD,1,0)  #IS THAT RIGHT Diameter vs RADIUS???

    def getNodeInfo(self):
        self.numNodes = np.sum(self.ind)
        self.nodeInfo = np.empty((self.numNodes), dtype = object)
        self.nodeTable = -np.ones_like(self.ind,dtype=np.int64)
        ind = np.pad(self.ind,1)
        c = 0
        for i in range(1,ind.shape[0]-1):
            for j in range(1,ind.shape[1]-1):
                for k in range(1,ind.shape[2]-1):

                    if (ind[i,j,k] == 1):

                        boundary = False   ### Should Take Faces out of loop to remove ifs!
                        boundaryID = [0,0,0]
                        inlet = False
                        if (i < 3):
                            boundary = True
                            boundaryID[0] = -1
                            if (self.subDomain.boundaryID[0] == -1 and self.inlet[0] == -1):
                                inlet = True

                        elif (i >= ind.shape[0]-3):
                            boundary = True
                            boundaryID[0] = 1
                            if (self.subDomain.boundaryID[0] == 1 and self.inlet[0] == 1):
                                inlet = True

                        if (j < 3):
                            boundary = True
                            boundaryID[1] = -1
                            if (self.subDomain.boundaryID[1] == -1 and self.inlet[1] == -1):
                                inlet = True

                        elif (j >= ind.shape[1]-3):
                            boundary = True
                            boundaryID[1] = 1
                            if (self.subDomain.boundaryID[1] == 1 and self.inlet[1] == 1):
                                inlet = True

                        if (k < 3):
                            boundary = True
                            boundaryID[2] = -1
                            if (self.subDomain.boundaryID[2] == -1 and self.inlet[2] == -1):
                                inlet = True

                        elif(k >= ind.shape[2]-3):
                            boundary = True
                            boundaryID[2] = 1
                            if (self.subDomain.boundaryID[2] == 1 and self.inlet[2] == 1):
                                inlet = True


                        globIndex = [self.subDomain.indexStart[0]+i-1,self.subDomain.indexStart[1]+j-1,self.subDomain.indexStart[2]+k-1]
                        self.nodeInfo[c] = Node(ID=c, localIndex = [i-1,j-1,k-1], globalIndex = globIndex, boundary = boundary, boundaryID = boundaryID, inlet = inlet)
                        self.nodeTable[i-1,j-1,k-1] = c

                        for d in self.Orientation.directions:
                            ii = self.Orientation.directions[d]['ID'][0]
                            jj = self.Orientation.directions[d]['ID'][1]
                            kk = self.Orientation.directions[d]['ID'][2]
                            if (ind[i+ii,j+jj,k+kk] == 1):
                                self.nodeInfo[c].validDirection(d)

                        self.nodeInfo[c].getAvailDirections()
                        self.nodeInfo[c].saveDirection = self.nodeInfo[c].availDirection

                        c = c + 1

    def getDirection3D(self,ID):
        """
        Provides first available direction and updates nodeInfo so that direction is not longer available
        and updates total number of available directions
        """

        i = self.nodeInfo[ID].localIndex[0]
        j = self.nodeInfo[ID].localIndex[1]
        k = self.nodeInfo[ID].localIndex[2]

        if self.nodeInfo[ID].direction.any():
            d  = np.argmax(self.nodeInfo[ID].direction)

            ii = self.Orientation.directions[d]['ID'][0]
            jj = self.Orientation.directions[d]['ID'][1]
            kk = self.Orientation.directions[d]['ID'][2]

            oppDir = self.Orientation.directions[d]['oppIndex']
            returnCell = self.nodeTable[i+ii,j+jj,k+kk]

            self.nodeInfo[returnCell].direction[oppDir] = 0
            self.nodeInfo[ID].direction[d] = 0
            self.nodeInfo[ID].getAvailDirections()
        else:
            returnCell = -1

        return returnCell

    def getConnectedSets(self):
        """
        Connects the NxNxN (or NXN) nodes into connected sets.
        1. Inlet
        2. Outlet
        3. DeadEnd
        """
        self.Sets = [Set(localID = 0)]
        numNodes = 0
        setCount = 0

        ### Loop Through All Nodes
        for node in range(0,self.numNodes):
            queue=[node]

            if self.nodeInfo[queue[0]].visited:
                queue.pop(0)
            else:
                while queue:
                    ID = queue.pop(0)
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

                        while (currentNode.availDirection > 0):

                            nodeValue = self.getDirection3D(ID)
                            if (nodeValue > -1):
                                if self.nodeInfo[nodeValue].visited:
                                    pass
                                else:
                                    queue.append(nodeValue)
                                    numNodes = numNodes + 1
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
                    nOwn = 0
                    matched = False
                    while not matched and nOwn < numOwnNodes:
                        otherNodes = otherBD[nbProc]['NeighborProcID'][nbProc]['setID'][otherSetKeys[testSetKey]]['boundaryNodes']
                        otherInlet = otherBD[nbProc]['NeighborProcID'][nbProc]['setID'][otherSetKeys[testSetKey]]['inlet']
                        if ownNodes[nOwn] in otherNodes:
                            if (ownInlet or otherInlet):
                                inlet = 1
                            self.matchedSets.append([self.subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet])
                            c = c + 1
                            matched = True
                            matchedOut = True
                        nOwn = nOwn + 1
                    testSetKey = testSetKey + 1

                if not matchedOut:
                    print("Set Not Matched! Hmmm",self.subDomain.ID,nbProc,ownSet)

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
        self.nwp = np.ones_like(self.ind)

        if (self.subDomain.boundaryID[0] == -1 and self.inlet[0] == -1):
            self.nwp[0,:,:] = 0
        if (self.subDomain.boundaryID[0] == 1 and self.inlet[0] == 1):
            self.nwp[-1,:,:] = 0
        if (self.subDomain.boundaryID[1] == -1 and self.inlet[1] == -1):
            self.nwp[:,0,:] = 0
        if (self.subDomain.boundaryID[1] == 1 and self.inlet[1] == 1):
            self.nwp[:,-1,:] = 0
        if (self.subDomain.boundaryID[2] == -1 and self.inlet[2] == -1):
            self.nwp[:,:,0] = 0
        if (self.subDomain.boundaryID[2] == 1 and self.inlet[2] == 1):
            self.nwp[:,:,-1] = 0

        for s in self.Sets:
            if s.inlet:
                for node in s.nodes:
                    NWNodes.append(node.localIndex)

        for n in NWNodes:
            self.nwp[n[0],n[1],n[2]] = 0

        self.nwpDist = edt.edt3d(self.nwp, anisotropy=(self.Domain.dX, self.Domain.dY, self.Domain.dZ))
        self.nwpFinal = np.where( (self.nwpDist <  self.probeD) & (self.subDomain.grid == 1),1,0)

    def drainCOMM(self):

        #### Faces ####
        reqs = [None]*self.Orientation.numFaces
        reqr = [None]*self.Orientation.numFaces
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborF[oppIndex]
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys()):
                reqs[fIndex] = comm.isend(self.boundaryData[self.subDomain.ID]['NeighborProcID'][oppNeigh],dest=oppNeigh)
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    reqr[fIndex] = comm.irecv(bytearray(1<<20),source=neigh)

        reqs = [i for i in reqs if i]
        MPI.Request.waitall(reqs)

        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.boundaryData:
                        self.boundaryData[neigh] = {'NeighborProcID':{}}
                    self.boundaryData[neigh]['NeighborProcID'][neigh] = reqr[fIndex].wait()

        #### Edges ####
        reqs = [None]*self.Orientation.numEdges
        reqr = [None]*self.Orientation.numEdges
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborE[oppIndex]
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys()):
                reqs[eIndex] = comm.isend(self.boundaryData[self.subDomain.ID]['NeighborProcID'][oppNeigh],dest=oppNeigh)
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    reqr[eIndex] = comm.irecv(bytearray(1<<20),source=neigh)

        reqs = [i for i in reqs if i]
        MPI.Request.waitall(reqs)

        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.boundaryData:
                        self.boundaryData[neigh] = {'NeighborProcID':{}}
                    self.boundaryData[neigh]['NeighborProcID'][neigh] = reqr[eIndex].wait()


        #### Corners ####
        reqs = [None]*self.Orientation.numCorners
        reqr = [None]*self.Orientation.numCorners
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = self.Orientation.corners[cIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborC[oppIndex]
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys()):
                reqs[cIndex] = comm.isend(self.boundaryData[self.subDomain.ID]['NeighborProcID'][oppNeigh],dest=oppNeigh)
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    reqr[cIndex] = comm.irecv(bytearray(1<<20),source=neigh)

        reqs = [i for i in reqs if i]
        MPI.Request.waitall(reqs)

        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.boundaryData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.boundaryData:
                        self.boundaryData[neigh] = {'NeighborProcID':{}}
                    self.boundaryData[neigh]['NeighborProcID'][neigh] = reqr[cIndex].wait()



def calcDrainage(rank,size,domain,subDomain,inlet,EDT):

    #pc = np.linspace(400, 400, 1)
    drain = Drainage(Domain = domain, Orientation = subDomain.Orientation, subDomain = subDomain, edt = EDT, gamma = 1., inlet = inlet)
    drain.getDiameter(250)
    drain.probeDistance()
    drain.getNodeInfo()
    drain.getConnectedSets()
    drain.getBoundarySets()
    drain.drainCOMM()
    drain.correctBoundarySets()

    ### Pass All matchedSets to proc 0 for global numbering and get local Set Counts for local GlobalID
    ### Add Column for Inlet True(1) or False(0)
    drainData = [drain.matchedSets,drain.setCount,drain.boundSetCount]
    drainData = comm.gather(drainData, root=0)
    if rank == 0:
        allMatchedSets = np.zeros([0,5],dtype=np.int64)
        numSets = np.zeros(size,dtype=np.int64)
        numBoundSets = np.zeros(size,dtype=np.int64)

        for n in range(0,size):
            allMatchedSets = np.append(allMatchedSets,drainData[n][0],axis=0)
            numSets[n] = drainData[n][1]
            numBoundSets[n] = drainData[n][2]

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

    drain.globalIndexStart = comm.scatter(localSetStart, root=0)
    drain.globalBoundarySetID = comm.scatter(globalSetScatter, root=0)

    drain.updateSetID()
    drain.getNWP()
