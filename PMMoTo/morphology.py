import numpy as np
from mpi4py import MPI
import math
from scipy import ndimage
comm = MPI.COMM_WORLD
import communication

class Morphology(object):
    def __init__(self,Domain,subDomain,grid,radius):
        self.Domain = Domain
        self.subDomain = subDomain
        self.Orientation = subDomain.Orientation
        self.structElem = None
        self.stuctRatio = np.zeros(3)
        self.grid = np.copy(grid)
        self.gridOut = np.copy(grid)
        self.radius = radius

    def genStructElem(self):

        self.structRatio = np.array([math.ceil(self.radius/self.Domain.dX),
                                     math.ceil(self.radius/self.Domain.dY),
                                     math.ceil(self.radius/self.Domain.dZ)],dtype=np.int64)

        x = np.linspace(-self.structRatio[0]*self.Domain.dX,self.structRatio[0]*self.Domain.dX,self.structRatio[0]*2+1)
        y = np.linspace(-self.structRatio[1]*self.Domain.dY,self.structRatio[1]*self.Domain.dY,self.structRatio[1]*2+1)
        z = np.linspace(-self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*2+1)

        xg,yg,zg = np.meshgrid(x,y,z,indexing='ij')
        s = xg**2 + yg**2 + zg**2

        self.structElem = np.array(s <= self.radius * self.radius)

    def haloCommPack(self):

        self.halo = np.zeros([6],dtype=np.int64)
        self.haloData = {self.subDomain.ID: {'NeighborProcID':{}}}
        self.Orientation.getSendSlices(self.structRatio,self.subDomain.buffer)

        self.slices = self.Orientation.sendFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            fID = self.Orientation.faces[fIndex]['ID']
            if neigh > -1:
                self.halo[fIndex]= np.max(np.abs(fID*self.structRatio))
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = self.grid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]]

        self.slices = self.Orientation.sendESlices
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            if neigh > -1:
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = self.grid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]]

        self.slices = self.Orientation.sendCSlices
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            if neigh > -1:
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = self.grid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]]

        self.haloGrid = np.pad(self.grid, ( (self.halo[1], self.halo[0]), (self.halo[3], self.halo[2]), (self.halo[5], self.halo[4]) ), 'constant', constant_values=255)

    def haloComm(self):

        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = communication.subDomainComm(self.Orientation,self.subDomain,self.haloData[self.subDomain.ID]['NeighborProcID'])

    def haloCommUnpack(self):
        self.Orientation.getRecieveSlices(self.structRatio,self.halo,self.haloGrid)

        #### Faces ####
        self.slices = self.Orientation.recvFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvFace[fIndex]
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
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvEdge[eIndex]
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
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvCorner[cIndex]
                    self.haloGrid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

    def morphAdd(self):
        gridOut = ndimage.binary_dilation(self.haloGrid,structure=self.structElem)
        print(self.structElem.shape)
        dim = gridOut.shape
        self.gridOut = gridOut[self.halo[1]:dim[0]-self.halo[0],
                               self.halo[3]:dim[1]-self.halo[2],
                               self.halo[5]:dim[2]-self.halo[4]]



def morph(rank,Domain,subDomain,grid,radius):
    sDMorph = Morphology(Domain = Domain,subDomain = subDomain, grid = grid, radius = radius)
    sDMorph.genStructElem()
    sDMorph.haloCommPack()
    sDMorph.haloComm()
    sDMorph.haloCommUnpack()
    sDMorph.morphAdd()
    return sDMorph
