import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

def subDomainComm(Orientation,subDomain,sendData):

    #### FACE ####
    reqs = [None]*Orientation.numFaces
    reqr = [None]*Orientation.numFaces
    recvDataFace = [None]*Orientation.numFaces
    for fIndex in Orientation.faces:
        neigh = subDomain.neighborF[fIndex]
        oppIndex = Orientation.faces[fIndex]['oppIndex']
        oppNeigh = subDomain.neighborF[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[fIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[fIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for fIndex in Orientation.faces:
        orientID = Orientation.faces[fIndex]['ID']
        neigh = subDomain.neighborF[fIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataFace[fIndex] = reqr[fIndex]#.wait()

    #### EDGES ####
    reqs = [None]*Orientation.numEdges
    reqr = [None]*Orientation.numEdges
    recvDataEdge = [None]*Orientation.numEdges
    for eIndex in Orientation.edges:
        neigh = subDomain.neighborE[eIndex]
        oppIndex = Orientation.edges[eIndex]['oppIndex']
        oppNeigh = subDomain.neighborE[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[eIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[eIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for eIndex in Orientation.edges:
        neigh = subDomain.neighborE[eIndex]
        oppIndex = Orientation.edges[eIndex]['oppIndex']
        oppNeigh = subDomain.neighborE[oppIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataEdge[eIndex] = reqr[eIndex]#.wait()

    #### CORNERS ####
    reqs = [None]*Orientation.numCorners
    reqr = [None]*Orientation.numCorners
    recvDataCorner = [None]*Orientation.numCorners
    for cIndex in Orientation.corners:
        neigh = subDomain.neighborC[cIndex]
        oppIndex = Orientation.corners[cIndex]['oppIndex']
        oppNeigh = subDomain.neighborC[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[cIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[cIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for cIndex in Orientation.corners:
        neigh = subDomain.neighborC[cIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataCorner[cIndex] = reqr[cIndex]#.wait()

    return recvDataFace,recvDataEdge,recvDataCorner


class Comm(object):
    def __init__(self,Domain,subDomain,grid):
        self.Domain = Domain
        self.subDomain = subDomain
        self.Orientation = subDomain.Orientation
        self.grid = grid
        self.halo = np.zeros([6],dtype=np.int64)
        self.haloData = {self.subDomain.ID: {'NeighborProcID':{}}}

    def haloCommPack(self,size):

        self.Orientation.getSendSlices(size,self.subDomain.buffer)

        self.slices = self.Orientation.sendFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            fID = self.Orientation.faces[fIndex]['ID']
            if neigh > -1:
                self.halo[fIndex]= np.max(np.abs(fID*size))
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][fIndex] = self.grid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]]


        self.slices = self.Orientation.sendESlices
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][eIndex] = self.grid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]]

        self.slices = self.Orientation.sendCSlices
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][cIndex] = self.grid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]]

        self.haloGrid = np.pad(self.grid, ( (self.halo[1], self.halo[0]), (self.halo[3], self.halo[2]), (self.halo[5], self.halo[4]) ), 'constant', constant_values=255)
        print(self.subDomain.ID,self.grid.shape,self.haloGrid.shape)

    def haloComm(self):
        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = subDomainComm(self.Orientation,self.subDomain,self.haloData[self.subDomain.ID]['NeighborProcID'])

    def haloCommUnpack(self,size):
        self.Orientation.getRecieveSlices(size,self.halo,self.haloGrid)

        #### Faces ####
        self.slices = self.Orientation.recvFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvFace[fIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Edges ####
        self.slices = self.Orientation.recvESlices
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvEdge[eIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Corners ####
        self.slices = self.Orientation.recvCSlices
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = self.Orientation.corners[cIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvCorner[cIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

    def haloCommunication(self,size):
        self.haloCommPack(size)
        self.haloComm()
        self.haloCommUnpack(size)
        return self.haloGrid,self.halo
