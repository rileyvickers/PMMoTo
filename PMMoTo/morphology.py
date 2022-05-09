import numpy as np
from mpi4py import MPI
import math
comm = MPI.COMM_WORLD


class Morphology(object):
    def __init__(self,Domain,subDomain):
        self.Domain = Domain
        self.subDomain = subDomain
        self.Orientation = subDomain.Orientation
        self.structElem = None
        self.stuctRatio = np.zeros(3)

    def genStructElem(self,radius):

        self.structRatio = np.array([math.ceil(radius/self.Domain.dX),
                                     math.ceil(radius/self.Domain.dY),
                                     math.ceil(radius/self.Domain.dZ)],dtype=np.int64)

        x = np.linspace(-self.structRatio[0]*self.Domain.dX,self.structRatio[0]*self.Domain.dX,self.structRatio[0]*2+1)
        y = np.linspace(-self.structRatio[1]*self.Domain.dY,self.structRatio[1]*self.Domain.dY,self.structRatio[1]*2+1)
        z = np.linspace(-self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*2+1)

        xg,yg,zg = np.meshgrid(x,y,z,indexing='ij')
        s = xg**2 + yg**2 + zg**2

        self.structElem = np.array(s <= radius * radius)

    def genHalo(self):

#### NEED TO SORT OUT BUFFER!!!!!
### Calculate slices for every fIndex and not just neigh > -1
        self.pad = np.zeros([6],dtype=np.int64)
        self.haloData = {self.subDomain.ID: {'NeighborProcID':{}}}
        self.Orientation.getSendSlices(self.structRatio)
        self.slices = self.Orientation.sendSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            fID = self.Orientation.faces[fIndex]['ID']
            if neigh > -1:
                self.pad[fIndex]= np.max(np.abs(fID*self.structRatio))
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = self.subDomain.grid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]]
        self.haloGrid = np.pad(self.subDomain.grid, ( (self.pad[1], self.pad[0]), (self.pad[3], self.pad[2]), (self.pad[5], self.pad[4]) ), 'constant', constant_values=255)



    def haloComm(self):

        #### Faces ####
        reqs = [None]*self.Orientation.numFaces
        reqr = [None]*self.Orientation.numFaces
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborF[oppIndex]
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys()):
                reqs[fIndex] = comm.isend(self.haloData[self.subDomain.ID]['NeighborProcID'][oppNeigh],dest=oppNeigh)
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    reqr[fIndex] = comm.irecv(bytearray(1<<20),source=neigh)

        reqs = [i for i in reqs if i]
        MPI.Request.waitall(reqs)


        self.Orientation.getRecieveSlices(self.pad,self.haloGrid)
        self.slices = self.Orientation.recvSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = reqr[fIndex].wait()
                    self.haloGrid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]
                    if self.subDomain.ID == 0:
                        print(self.subDomain.ID,neigh,self.subDomain.grid.shape,self.haloGrid.shape,self.haloData[neigh]['NeighborProcID'][neigh].shape, \
                        self.pad,self.slices[fIndex])

def morph(rank,Domain,subDomain):
    sDMorph = Morphology(Domain = Domain,subDomain = subDomain)
    sDMorph.genStructElem(radius=Domain.dX*4)
    sDMorph.genHalo()
    sDMorph.haloComm()

    return sDMorph
