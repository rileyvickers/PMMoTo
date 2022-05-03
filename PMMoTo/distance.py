import numpy as np
from mpi4py import MPI
from scipy.spatial import KDTree
import edt
comm = MPI.COMM_WORLD

""" Solid = 0, Pore = 1 """

""" TO DO:
           Generate Distance Stastics
           Get Load Balacing Statistics
           Cython
           Switch to pass peridic info and not generate from samples??
           Redo Domain decomposition - Maybe
"""

class EDT(object):

    def __init__(self,ID,subDomain,Domain,Orientation,grid):
        bufferSize         = 1
        self.extendFactor  = 0.7
        self.useIndex      = False
        self.ID            = ID
        self.subDomain     = subDomain
        self.Domain        = Domain
        self.Orientation   = Orientation
        self.EDT = np.zeros_like(self.subDomain.grid)
        self.solids = None
        self.nS = 0
        self.trimmedSolids = None
        self.solidsAll = {self.ID: {'orientID':{}}}
        self.edgeSolids = []
        self.cornerSolids = []

        self.grid = grid
        self.x = self.subDomain.x
        self.y = self.subDomain.y
        self.z = self.subDomain.z
        self.subDomainSize = self.subDomain.subDomainSize
        self.buffer = self.subDomain.buffer
        self.distVals = None
        self.distCounts  = None

    def genLocalEDT(self,):
        self.EDT = edt.edt3d(self.grid, anisotropy=(self.Domain.dX, self.Domain.dY, self.Domain.dZ))

    def getBoundarySolids(self):
        area = 2*self.grid.shape[0]*self.grid.shape[1] + 2*self.grid.shape[0]*self.grid.shape[2] + 2*self.grid.shape[1]*self.grid.shape[2]
        self.solids = -np.ones([area,4],dtype='int32')

        for faceID in self.Orientation.faces:
            order  = [None]*3
            nC  = self.Orientation.faces[faceID]['nC']
            nM  = self.Orientation.faces[faceID]['nM']
            nN  = self.Orientation.faces[faceID]['nN']
            dir = self.Orientation.faces[faceID]['dir']


            if (dir == 1):
                for m in range(0,self.grid.shape[nM]):
                    for n in range(0,self.grid.shape[nN]):
                        solid = False
                        c = 0
                        while not solid and c < self.grid.shape[nC]:
                            order[nC] = c
                            order[nM] = m
                            order[nN] = n
                            if self.grid[order[0],order[1],order[2]] == 0:
                                solid = True
                                self.solids[self.nS,0:3] = order
                                self.solids[self.nS,3] = faceID
                                self.nS = self.nS + 1
                            else:
                                c = c + 1
                        if (not solid and c == self.grid.shape[nC]):
                            order[nC] = -1
                            self.solids[self.nS,0:3] = order
                            self.solids[self.nS,3] = faceID
                            self.nS = self.nS + 1

            elif (dir == -1):
                for m in range(0,self.grid.shape[nM]):
                    for n in range(0,self.grid.shape[nN]):
                        solid = False
                        c = self.grid.shape[nC] - 1
                        while not solid and c > 0:
                            order[nC] = c
                            order[nM] = m
                            order[nN] = n
                            if self.grid[order[0],order[1],order[2]] == 0:
                                solid = True
                                self.solids[self.nS,0:3] = order
                                self.solids[self.nS,3] = faceID
                                self.nS = self.nS + 1
                            else:
                                c = c - 1
                        if (not solid and c == 0):
                            order[nC] = -1
                            self.solids[self.nS,0:3] = order
                            self.solids[self.nS,3] = faceID
                            self.nS = self.nS + 1

    def trimBoundarySolids(self):
        """
        TODO: Trim Edge duplicates!!
        Trim to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
        """
        self.trimmedSolids = [[] for _ in range(len(self.Orientation.faces))]
        extend = [self.extendFactor*x for x in self.subDomainSize]
        points = []
        pointsXYZ = []
        points = self.solids[np.where( (self.solids[:,0]>-1) & (self.solids[:,1]>-1) & (self.solids[:,2]>-1) )][:,0:3].tolist()
        points = np.unique(np.array(points),axis=0).tolist()

        minAdd = min(self.Domain.dX,self.Domain.dY,self.Domain.dZ)

        for x,y,z in points:
            pointsXYZ.append([self.x[x],self.y[y],self.z[z]] )
        tree = KDTree(pointsXYZ)

        for faceID in self.Orientation.faces:
            maxNeigh = 8
            order  = [None]*3
            nC  = self.Orientation.faces[faceID]['nC']
            nM  = self.Orientation.faces[faceID]['nM']
            nN  = self.Orientation.faces[faceID]['nN']
            dir = self.Orientation.faces[faceID]['dir']

            if (dir == 1):
                for m in range(0,self.grid.shape[nM]):
                    for n in range(0,self.grid.shape[nN]):
                        c = self.buffer[nC][0]
                        order[nC] = c
                        order[nM] = m
                        order[nN] = n
                        if(self.grid[order[0],order[1],order[2]] == 1):
                            maxD = self.EDT[order[0],order[1],order[2]]+minAdd
                            d,ind = tree.query([self.x[order[0]],self.y[order[1]],self.z[order[2]]],distance_upper_bound=maxD,k=maxNeigh)
                            if (d[0] < maxD):
                                self.trimmedSolids[faceID].append([self.x[points[ind[0]][0]],self.y[points[ind[0]][1]],self.z[points[ind[0]][2]],points[ind[0]][0],points[ind[0]][1],points[ind[0]][2]])
                                nNeigh = 1
                                while (nNeigh < maxNeigh and d[nNeigh]==d[nNeigh-1]):
                                    self.trimmedSolids[faceID].append([self.x[points[ind[nNeigh]][0]],self.y[points[ind[nNeigh]][1]],self.z[points[ind[nNeigh]][2]],points[ind[nNeigh]][0],points[ind[nNeigh]][1],points[ind[nNeigh]][2]])
                                    nNeigh = nNeigh + 1
                                if (nNeigh == maxNeigh):
                                    print("WARNING: All solids may not be incldued resulting in errors")
                        else:
                            self.trimmedSolids[faceID].append([self.x[order[0]],self.y[order[1]],self.z[order[2]],order[0],order[1],order[2]])

            elif(dir == -1):
                for m in range(0,self.grid.shape[nM]):
                    for n in range(0,self.grid.shape[nN]):
                        c = self.grid.shape[nC] - 1 - self.buffer[nC][1]
                        order[nC] = c
                        order[nM] = m
                        order[nN] = n
                        if(self.grid[order[0],order[1],order[2]] == 1):
                            maxD = self.EDT[order[0],order[1],order[2]]+minAdd
                            d,ind = tree.query([self.x[order[0]],self.y[order[1]],self.z[order[2]]],distance_upper_bound=maxD,k=maxNeigh)
                            if (d[0] < maxD):
                                self.trimmedSolids[faceID].append([self.x[points[ind[0]][0]],self.y[points[ind[0]][1]],self.z[points[ind[0]][2]],points[ind[0]][0],points[ind[0]][1],points[ind[0]][2]])
                                nNeigh = 1
                                while (nNeigh < maxNeigh and d[nNeigh]==d[nNeigh-1]):
                                    self.trimmedSolids[faceID].append([self.x[points[ind[nNeigh]][0]],self.y[points[ind[nNeigh]][1]],self.z[points[ind[nNeigh]][2]],points[ind[nNeigh]][0],points[ind[nNeigh]][1],points[ind[nNeigh]][2]])
                                    nNeigh = nNeigh + 1
                                if (nNeigh == maxNeigh):
                                    print("WARNING: All solids may not be incldued resulting in errors")
                        else:
                            self.trimmedSolids[faceID].append([self.x[order[0]],self.y[order[1]],self.z[order[2]],order[0],order[1],order[2]])

            self.trimmedSolids[faceID] = np.array(self.trimmedSolids[faceID])
            self.trimmedSolids[faceID] = self.trimmedSolids[faceID][:,0:3]
            name = self.Orientation.faces[faceID]['ID']
            self.solidsAll[self.ID]['orientID'][name] = np.copy(self.trimmedSolids[faceID])

    def getEdgeSolids(self):

        self.edgeSolids = [[] for _ in range(len(self.Orientation.edges))]
        for eIndex in self.Orientation.edges:
            edgeID = self.Orientation.edges[eIndex]['ID']
            face1 = self.trimmedSolids[self.Orientation.edges[eIndex]['faceIndex'][0]]
            face2 = self.trimmedSolids[self.Orientation.edges[eIndex]['faceIndex'][1]]
            arg1  = self.Orientation.edges[eIndex]['dir'][0]
            arg2  = self.Orientation.edges[eIndex]['dir'][1]
            coords = [self.x,self.y,self.z]
            coord1 = coords[arg1]
            coord2 = coords[arg2]

            if self.useIndex:
                extend = [self.extendFactor*x for x in self.nodes]
                plusArg = 3
                if edgeID[arg1] == 1:
                    dom11 = self.nodes[arg1] - extend[arg1]
                    dom12 = self.nodes[arg1]
                elif (edgeID[arg1] == -1):
                    dom11 = 0
                    dom12 = extend[arg1]
                if edgeID[arg2] == 1:
                    dom21 = self.nodes[arg2] - extend[arg2]
                    dom22 = self.nodes[arg2]
                elif (edgeID[arg2] == -1):
                    dom21 = 0
                    dom22 = extend[arg2]
            else:
                extend = [self.extendFactor*x for x in self.subDomainSize]
                plusArg = 0
                if edgeID[arg1] == 1:
                    dom11 = coord1[-1] - extend[arg1]
                    dom12 = coord1[-1]
                elif (edgeID[arg1] == -1):
                    dom11 = coord1[0]
                    dom12 = coord1[0] + extend[arg1]
                if edgeID[arg2] == 1:
                    dom21 = coord2[-1] - extend[arg2]
                    dom22 = coord2[-1]
                elif (edgeID[arg2] == -1):
                    dom21 = coord2[0]
                    dom22 = coord2[0] + extend[arg2]

            truth1 = dom21 <= face1[:,arg2+plusArg]
            truth2 = face1[:,arg2+plusArg] <= dom22
            self.edgeSolids[eIndex] = face1[np.where( truth1 & truth2 )]
            truth1 = dom11 <= face2[:,arg1+plusArg]
            truth2 = face2[:,arg1+plusArg] <= dom12
            self.edgeSolids[eIndex] = np.append(self.edgeSolids[eIndex],face2[np.where( truth1 & truth2 )],axis=0)
            self.edgeSolids[eIndex] = np.unique(np.array(self.edgeSolids[eIndex]),axis=0)

    def getCornerSolids(self):

        self.cornerSolids = [[] for _ in range(len(self.Orientation.corners))]

        for cIndex in self.Orientation.corners:
            cID = self.Orientation.corners[cIndex]['ID']
            face1 = self.trimmedSolids[self.Orientation.corners[cIndex]['faceIndex'][0]]
            face2 = self.trimmedSolids[self.Orientation.corners[cIndex]['faceIndex'][1]]
            face3 = self.trimmedSolids[self.Orientation.corners[cIndex]['faceIndex'][2]]
            if self.useIndex:
                plusArg = 3
                extend = [self.extendFactor*x for x in self.nodes]

                if (cID[0] == 1):
                    dom11 = self.nodes[0] - extend[0]
                    dom12 = self.nodes[0]
                elif (cID[0] == -1):
                    dom11 = 0
                    dom12 = extend[0]
                if (cID[1] == 1):
                    dom21 = self.nodes[1] - extend[1]
                    dom22 = self.nodes[1]
                elif (cID[1] == -1):
                    dom21 = 0
                    dom22 = extend[1]
                if (cID[2] == 1):
                    dom31 = self.nodes[2] - extend[2]
                    dom32 = self.nodes[2]
                elif (cID[2] == -1):
                    dom31 = 0
                    dom32 = extend[2]

            else:
                plusArg = 0
                extend = [self.extendFactor*x for x in self.subDomainSize]
                if (cID[0] == 1):
                    dom11 = self.x[-1] - extend[0]
                    dom12 = self.x[-1]
                elif (cID[0] == -1):
                    dom11 = self.x[0]
                    dom12 = self.x[0] + extend[0]

                if (cID[1] == 1):
                    dom21 = self.y[-1] - extend[1]
                    dom22 = self.y[-1]
                elif (cID[1] == -1):
                    dom21 = self.y[0]
                    dom22 = self.y[0] + extend[1]
                if (cID[2] == 1):
                    dom31 = self.z[-1] - extend[2]
                    dom32 = self.z[-1]
                elif (cID[2] == -1):
                    dom31 = self.z[0]
                    dom32 = self.z[0] + extend[2]

            truth1 = (dom21 <= face1[:,1+plusArg]) & (face1[:,1+plusArg] <= dom22)
            truth2 = (dom31 <= face1[:,2+plusArg]) & (face1[:,2+plusArg] <= dom32)
            self.cornerSolids[cIndex] = face1[np.where(truth1 & truth2)]
            truth1 = (dom11 <= face2[:,0+plusArg]) & (face2[:,0+plusArg] <= dom12)
            truth2 = (dom31 <= face2[:,2+plusArg]) & (face2[:,2+plusArg] <= dom32)
            self.cornerSolids[cIndex] = np.append(self.cornerSolids[cIndex],face2[np.where(truth1 & truth2)],axis=0)
            truth1 = (dom11 <= face3[:,0+plusArg]) & (face3[:,0+plusArg] <= dom12)
            truth2 = (dom21 <= face3[:,1+plusArg]) & (face3[:,1+plusArg] <= dom22)
            self.cornerSolids[cIndex] = np.append(self.cornerSolids[cIndex], face3[np.where(truth1 & truth2)],axis=0)

    def fixInterfaceCalc(self,tree,faceID):

        order  = [None]*3
        orderL = [None]*3
        nC  = self.Orientation.faces[faceID]['nC']
        nM  = self.Orientation.faces[faceID]['nM']
        nN  = self.Orientation.faces[faceID]['nN']
        dir = self.Orientation.faces[faceID]['dir']
        coords = [self.x,self.y,self.z]
        minD = min(self.Domain.dX,self.Domain.dY,self.Domain.dZ)

        faceSolids = self.solids[np.where(self.solids[:,3]==faceID)][:,0:3]

        if (dir == 1):
            for i in range(0,faceSolids.shape[0]):

                if faceSolids[i,nC] < 0:
                    endL = self.grid.shape[nC]
                else:
                    endL = faceSolids[i,nC]

                distChanged = True
                l = 0
                while distChanged and l < endL:
                    cL = l
                    cM = faceSolids[i,nM]
                    cN = faceSolids[i,nN]

                    m = cM
                    n = cN

                    order[nC] = coords[nC][cL]
                    order[nM] = coords[nM][cM]
                    order[nN] = coords[nN][cN]
                    orderL[nC] = l
                    orderL[nM] = m
                    orderL[nN] = n

                    maxD = self.EDT[orderL[0],orderL[1],orderL[2]]
                    if (maxD > minD):
                        d,ind = tree.query([order],p=2,distance_upper_bound=maxD)
                        if d < maxD:
                            self.EDT[orderL[0],orderL[1],orderL[2]] = d
                            distChanged = True
                    l = l + 1

        if (dir == -1):
            for i in range(0,faceSolids.shape[0]):

                if faceSolids[i,nC] < 0:
                    endL = 0
                else:
                    endL = faceSolids[i,nC]

                distChanged = True
                l = self.grid.shape[nC] - 1

                while distChanged and l > endL:
                    cL = l
                    cM = faceSolids[i,nM]
                    cN = faceSolids[i,nN]
                    m = cM
                    n = cN
                    order[nC] = coords[nC][cL]
                    order[nM] = coords[nM][cM]
                    order[nN] = coords[nN][cN]
                    orderL[nC] = l
                    orderL[nM] = m
                    orderL[nN] = n

                    maxD = self.EDT[orderL[0],orderL[1],orderL[2]]
                    if (maxD > minD):
                        d,ind = tree.query([order],p=2,distance_upper_bound=maxD)
                        if d < maxD:
                            self.EDT[orderL[0],orderL[1],orderL[2]] = d
                            distChanged = True
                    l = l - 1

    def initRecieve(self):

        for neigh in self.subDomain.neighborF:
            if neigh > -1 and neigh != self.ID:
                self.solidsAll[neigh] = {'orientID':{}}

        for neigh in self.subDomain.neighborE:
            if neigh > -1 and neigh != self.ID:
                self.solidsAll[neigh] = {'orientID':{}}

        for neigh in self.subDomain.neighborC:
            if neigh > -1 and neigh != self.ID:
                self.solidsAll[neigh] = {'orientID':{}}

    def fixInterface(self):
        for fIndex in self.Orientation.faces:
            orientID = self.Orientation.faces[fIndex]['ID']
            data = np.empty((0,3))
            for procs in self.solidsAll.keys():
                for fID in self.solidsAll[procs]['orientID'].keys():
                    if fID == orientID:
                        data = np.append(data,self.solidsAll[procs]['orientID'][fID],axis=0)
            tree = KDTree(data)
            self.fixInterfaceCalc(tree,fIndex)

    def EDTComm(self):

        #### FACE ####
        reqs = [None]*self.Orientation.numFaces
        reqr = [None]*self.Orientation.numFaces
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborF[oppIndex]
            if (oppNeigh > -1 and neigh != self.ID):
                reqs[fIndex] = comm.isend(self.trimmedSolids[oppIndex],dest=oppNeigh)
            if (neigh > -1 and neigh != self.ID):
                reqr[fIndex] = comm.irecv(bytearray(1<<20),source=neigh)


        reqs = [i for i in reqs if i]
        MPI.Request.waitall(reqs)

        for fIndex in self.Orientation.faces:
            orientID = self.Orientation.faces[fIndex]['ID']
            neigh = self.subDomain.neighborF[fIndex]
            perFace  = self.subDomain.neighborPerF[fIndex]
            if (neigh > -1 and neigh != self.ID):
                self.solidsAll[neigh]['orientID'][orientID] = reqr[fIndex].wait()
                if (perFace.any() != 0):
                    perCorrection = perFace*self.Domain.domainLength
                    self.solidsAll[neigh]['orientID'][orientID] = self.solidsAll[neigh]['orientID'][orientID]-perCorrection
            elif (neigh == self.ID):
                oppIndex = self.Orientation.faces[fIndex]['oppIndex']
                orientID = self.Orientation.faces[fIndex]['ID']
                perCorrection = perFace*self.Domain.domainLength
                self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.trimmedSolids[oppIndex]-perCorrection,axis=0)

        # #### EDGES ####
        reqs = [None]*self.Orientation.numEdges
        reqr = [None]*self.Orientation.numEdges
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborE[oppIndex]
            if (oppNeigh > -1 and neigh != self.ID):
                reqs[eIndex] = comm.isend(self.edgeSolids[oppIndex],dest=oppNeigh)
            if (neigh > -1 and neigh != self.ID):
                reqr[eIndex] = comm.irecv(bytearray(1<<20),source=neigh)

        reqs = [i for i in reqs if i]
        MPI.Request.waitall(reqs)

        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            perEdge = self.subDomain.neighborPerE[eIndex]
            if (neigh > -1  and neigh != self.ID):
                faceIndex = self.Orientation.edges[eIndex]['faceIndex']
                data = reqr[eIndex].wait()
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if orientID in self.solidsAll[neigh]['orientID']:
                        if (perEdge.any() != 0):
                            perCorrection = perEdge*self.Domain.domainLength
                            self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],data - perCorrection,axis=0)
                        else:
                            self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],data,axis=0)
                    else:
                        if (perEdge.any() != 0):
                            perCorrection = perEdge*self.Domain.domainLength
                            self.solidsAll[neigh]['orientID'][orientID] = data - perCorrection
                        else:
                            self.solidsAll[neigh]['orientID'][orientID] = data

            if(neigh == self.ID):
                faceIndex = self.Orientation.edges[eIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if (perEdge.any() != 0):
                        perCorrection = perEdge*self.Domain.domainLength
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.edgeSolids[oppIndex]-perCorrection,axis=0)
                    else:
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.edgeSolids[oppIndex],axis=0)

        #### CORNERS ####
        reqs = [None]*self.Orientation.numCorners
        reqr = [None]*self.Orientation.numCorners
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = self.Orientation.corners[cIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborC[oppIndex]
            if (oppNeigh > -1 and neigh != self.ID):
                reqs[cIndex] = comm.isend(self.cornerSolids[oppIndex],dest=oppNeigh)
            if (neigh > -1 and neigh != self.ID):
                reqr[cIndex] = comm.irecv(bytearray(1<<20),source=neigh)

        reqs = [i for i in reqs if i]
        MPI.Request.waitall(reqs)

        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            perCorner = self.subDomain.neighborPerC[cIndex]
            if (neigh > -1 and neigh != self.ID):
                data = reqr[cIndex].wait()
                faceIndex = self.Orientation.corners[cIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if orientID in self.solidsAll[neigh]['orientID']:
                        if (perCorner.any() != 0):
                            perCorrection = perCorner*self.Domain.domainLength*periodic
                            self.solidsAll[neigh]['orientID'][orientID] =np.append(self.solidsAll[neigh]['orientID'][orientID],data-perCorrection,axis=0)
                        else:
                            self.solidsAll[neigh]['orientID'][orientID] =np.append(self.solidsAll[neigh]['orientID'][orientID],data,axis=0)
                    else:
                        self.solidsAll[neigh]['orientID'][orientID] = data
                        if (perCorner.any() != 0):
                            perCorrection = perCorner*self.Domain.domainLength*periodic
                            self.solidsAll[neigh]['orientID'][orientID] = self.solidsAll[neigh]['orientID'][orientID]-perCorrection
            if neigh == self.ID:
                faceIndex = self.Orientation.corners[cIndex]['faceIndex']
                oppIndex = self.Orientation.corners[cIndex]['oppIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if (perCorner.any() != 0):
                        perCorrection = perCorner*self.Domain.domainLength*periodic
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.cornerSolids[oppIndex]-perCorrection,axis=0)
                    else:
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.cornerSolids[oppIndex],axis=0)

    def genStats(self):
        own = self.subDomain.ownNodes
        ownEDT =  self.EDT[own[0][0]:own[0][1],
                            own[1][0]:own[1][1],
                            own[2][0]:own[2][1]]
        self.distVals,self.distCounts  = np.unique(ownEDT,return_counts=True)

def calcEDT(rank,domain,subDomains,subDomain):
    numSubDomains = np.prod(subDomains)

    gridIn = subDomain.grid
    sDEDT = EDT(Domain = domain, ID = rank, subDomain = subDomain, Orientation = subDomain.Orientation, grid = gridIn)
    sDEDT.getBoundarySolids()
    sDEDT.genLocalEDT()
    sDEDT.trimBoundarySolids()
    sDEDT.getEdgeSolids()
    sDEDT.getCornerSolids()
    sDEDT.initRecieve()
    sDEDT.EDTComm()
    sDEDT.fixInterface()

    EDTStats = True
    if EDTStats:
        sDEDT.genStats()
        EDTData = [sDEDT.ID,sDEDT.distVals,sDEDT.distCounts]
        EDTData = comm.gather(EDTData, root=0)
        if rank == 0:
            bins = np.empty([])
            for d in EDTData:
                if d[0] == 0:
                    bins = d[1]
                else:
                    bins = np.append(bins,d[1],axis=0)
                bins = np.unique(bins)

            counts = np.zeros_like(bins,dtype=np.int64)
            for d in EDTData:
                for n in range(0,d[1].size):
                    ind = np.where(bins==d[1][n])[0][0]
                    counts[ind] = counts[ind] + d[2][n]

            stats = np.stack((bins,counts), axis = 1)
            print(stats)

    return sDEDT
