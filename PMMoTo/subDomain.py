import numpy as np
from mpi4py import MPI
from MAtools import domainGen
comm = MPI.COMM_WORLD

class Orientation(object):
    def __init__(self):
        self.numFaces = 6
        self.numEdges = 12
        self.numCorners = 8
        self.faces = {0:{'ID':(1,0,0),  'oppIndex':1, 'nC':0, 'nM':1, 'nN':2, 'dir':-1},
                      1:{'ID':(-1,0,0), 'oppIndex':0, 'nC':0, 'nM':1, 'nN':2, 'dir':1},
                      2:{'ID':(0,1,0),  'oppIndex':3, 'nC':1, 'nM':0, 'nN':2, 'dir':-1},
                      3:{'ID':(0,-1,0), 'oppIndex':2, 'nC':1, 'nM':0, 'nN':2, 'dir':1},
                      4:{'ID':(0,0,1),  'oppIndex':5, 'nC':2, 'nM':0, 'nN':1, 'dir':-1},
                      5:{'ID':(0,0,-1), 'oppIndex':4, 'nC':2, 'nM':0, 'nN':1, 'dir':1},
                      }
        self.edges = {0 :{'ID':(1,1,0),  'oppIndex':5, 'faceIndex':(0,2), 'dir':(0,1)},
                      1 :{'ID':(1,-1,0), 'oppIndex':4, 'faceIndex':(0,3), 'dir':(0,1)},
                      2 :{'ID':(1,0,1),  'oppIndex':7, 'faceIndex':(0,4), 'dir':(0,2)},
                      3 :{'ID':(1,0,-1), 'oppIndex':6, 'faceIndex':(0,5), 'dir':(0,2)},
                      4 :{'ID':(-1,1,0), 'oppIndex':1, 'faceIndex':(1,2), 'dir':(0,1)},
                      5 :{'ID':(-1,-1,0),'oppIndex':0, 'faceIndex':(1,3), 'dir':(0,1)},
                      6 :{'ID':(-1,0,1), 'oppIndex':3, 'faceIndex':(1,4), 'dir':(0,2)},
                      7 :{'ID':(-1,0,-1),'oppIndex':2, 'faceIndex':(1,5), 'dir':(0,2)},
                      8 :{'ID':(0,1, 1), 'oppIndex':11,'faceIndex':(2,4), 'dir':(1,2)},
                      9 :{'ID':(0,1,-1), 'oppIndex':10,'faceIndex':(2,5), 'dir':(1,2)},
                      10:{'ID':(0,-1,1), 'oppIndex':9, 'faceIndex':(3,4), 'dir':(1,2)},
                      11:{'ID':(0,-1,-1),'oppIndex':8, 'faceIndex':(3,5), 'dir':(1,2)},
                       }
        self.corners = {0:{'ID':(1,1,1),   'oppIndex':7, 'faceIndex':(0,2,4)},
                        1:{'ID':(1,1,-1),  'oppIndex':6, 'faceIndex':(0,2,5)},
                        2:{'ID':(1,-1,1),  'oppIndex':5, 'faceIndex':(0,3,4)},
                        3:{'ID':(1,-1,-1), 'oppIndex':4, 'faceIndex':(0,3,5)},
                        4:{'ID':(-1,1,1),  'oppIndex':3, 'faceIndex':(1,2,4)},
                        5:{'ID':(-1,1,-1), 'oppIndex':2, 'faceIndex':(1,2,5)},
                        6:{'ID':(-1,-1,1), 'oppIndex':1, 'faceIndex':(1,3,4)},
                        7:{'ID':(-1,-1,-1),'oppIndex':0, 'faceIndex':(1,3,5)},
                        }

class Domain(object):
    def __init__(self,nodes,domainSize,subDomains,periodic):
        self.nodes        = nodes
        self.domainSize   = domainSize
        self.periodic     = periodic
        self.subDomains   = subDomains
        self.subNodes     = np.zeros([3])
        self.subNodesRem  = np.zeros([3])
        self.domainLength = np.zeros([3])
        self.dX = 0
        self.dY = 0
        self.dZ = 0

    def getdXYZ(self):
        self.domainLength[0] = (self.domainSize[0,1]-self.domainSize[0,0])
        self.domainLength[1] = (self.domainSize[1,1]-self.domainSize[1,0])
        self.domainLength[2] = (self.domainSize[2,1]-self.domainSize[2,0])
        self.dX = self.domainLength[0]/self.nodes[0]
        self.dY = self.domainLength[1]/self.nodes[1]
        self.dZ = self.domainLength[2]/self.nodes[2]

    def getSubNodes(self):
        self.subNodes[0],self.subNodesRem[0] = divmod(self.nodes[0],self.subDomains[0])
        self.subNodes[1],self.subNodesRem[1] = divmod(self.nodes[1],self.subDomains[1])
        self.subNodes[2],self.subNodesRem[2] = divmod(self.nodes[2],self.subDomains[2])

class subDomain(object):
    def __init__(self,ID,subDomains,Domain,Orientation):
        bufferSize        = 1
        self.extendFactor = 0.7
        self.useIndex     = False
        self.ID          = ID
        self.subDomains  = subDomains
        self.Domain      = Domain
        self.Orientation = Orientation
        self.nodes       = np.zeros([3],dtype=np.int64)
        self.indexStart  = np.zeros([3],dtype=np.int64)
        self.subID       = np.zeros([3],dtype=np.int64)
        self.lookUpID    = np.zeros(subDomains,dtype=np.int64)
        self.buffer      = bufferSize*np.ones([3,2],dtype = np.int64)
        self.numSubDomains = np.prod(subDomains)
        self.neighborF    = -np.ones(6,dtype = np.int64)
        self.neighborE    = -np.ones(12,dtype = np.int64)
        self.neighborC    = -np.ones(8,dtype = np.int64)
        self.neighborPerF =  np.zeros([6,3],dtype = np.int64)
        self.neighborPerE =  np.zeros([12,3],dtype = np.int64)
        self.neighborPerC =  np.zeros([8,3],dtype = np.int64)
        self.ownNodes     = np.zeros([3,2],dtype = np.int64)
        self.ownNodesTotal= 0
        self.poreNodes    = 0
        self.subDomainSize = np.zeros([3,1])

    def getInfo(self):
        n = 0
        for i in range(0,self.subDomains[0]):
            for j in range(0,self.subDomains[1]):
                for k in range(0,self.subDomains[2]):
                    self.lookUpID[i,j,k] = n
                    if n == self.ID:
                        self.subID[0] = i
                        self.subID[1] = j
                        self.subID[2] = k
                        self.nodes[0] = self.Domain.subNodes[0]
                        self.nodes[1] = self.Domain.subNodes[1]
                        self.nodes[2] = self.Domain.subNodes[2]
                        self.indexStart[0] = self.Domain.subNodes[0]*i
                        self.indexStart[1] = self.Domain.subNodes[1]*j
                        self.indexStart[2] = self.Domain.subNodes[2]*k
                        if (i == self.subDomains[0]-1):
                            self.nodes[0] += self.Domain.subNodesRem[0]
                        if (j == self.subDomains[1]-1):
                            self.nodes[1] += self.Domain.subNodesRem[1]
                        if (k == self.subDomains[2]-1):
                            self.nodes[2] += self.Domain.subNodesRem[2]
                    n = n + 1

    def getXYZ(self):

        if (self.subID[0] == 0 and not self.Domain.periodic[0]):
            self.buffer[0][0] = 0
        if (self.subID[0] == (self.subDomains[0] - 1) and not self.Domain.periodic[0]):
            self.buffer[0][1] = 0
        if (self.subID[1] == 0 and not self.Domain.periodic[1]):
            self.buffer[1][0] = 0
        if (self.subID[1] == (self.subDomains[1] - 1) and not self.Domain.periodic[1]):
            self.buffer[1][1] = 0
        if (self.subID[2] == 0 and not self.Domain.periodic[2]):
            self.buffer[2][0] = 0
        if (self.subID[2] == (self.subDomains[2] - 1) and not self.Domain.periodic[2]):
            self.buffer[2][1] = 0

        self.x = np.zeros([self.nodes[0] + self.buffer[0][0] + self.buffer[0][1]])
        self.y = np.zeros([self.nodes[1] + self.buffer[1][0] + self.buffer[1][1]])
        self.z = np.zeros([self.nodes[2] + self.buffer[2][0] + self.buffer[2][1]])

        c = 0
        for i in range(-self.buffer[0][0], self.nodes[0] + self.buffer[0][1]):
            self.x[c] = (self.indexStart[0] + i)*self.Domain.dX + self.Domain.dX/2
            c = c + 1

        c = 0
        for j in range(-self.buffer[1][0], self.nodes[1] + self.buffer[1][1]):
            self.y[c] = (self.indexStart[1] + j)*self.Domain.dY + self.Domain.dY/2
            c = c + 1

        c = 0
        for k in range(-self.buffer[2][0], self.nodes[2] + self.buffer[2][1]):
            self.z[c] = (self.indexStart[2] + k)*self.Domain.dZ + self.Domain.dZ/2
            c = c + 1

        self.ownNodesTotal = self.nodes[0] * self.nodes[1] * self.nodes[2]

        self.ownNodes[0] = [self.buffer[0][0],self.nodes[0]+self.buffer[0][0]]
        self.ownNodes[1] = [self.buffer[1][0],self.nodes[1]+self.buffer[1][0]]
        self.ownNodes[2] = [self.buffer[2][0],self.nodes[2]+self.buffer[2][0]]

        self.nodes[0] = self.nodes[0] + self.buffer[0][0] + self.buffer[0][1]
        self.nodes[1] = self.nodes[1] + self.buffer[1][0] + self.buffer[1][1]
        self.nodes[2] = self.nodes[2] + self.buffer[2][0] + self.buffer[2][1]

        if self.buffer[0][0] != 0:
            self.indexStart[0] = self.indexStart[0] - self.buffer[0][0]

        if self.buffer[1][0] != 0:
            self.indexStart[1] = self.indexStart[1] - self.buffer[1][0]

        if self.buffer[2][0] != 0:
            self.indexStart[2] = self.indexStart[2] - self.buffer[2][0]


        self.subDomainSize = [self.x[-1] - self.x[0],
                              self.y[-1] - self.y[0],
                              self.z[-1] - self.z[0]]

    def genDomain(self,sphereData):
        self.visited = np.zeros([self.x.shape[0],self.y.shape[0],self.z.shape[0]],dtype=np.int8)
        self.grid = domainGen(self.x,self.y,self.z,sphereData)

        if (np.sum(self.grid) == np.prod(self.nodes)):
            print("This code requires at least 1 solid voxel in each subdomain. Please reorder processors!")
            sys.exit()

    def getNeighbors(self):
        """
        Get the Face,Edge, and Corner Neighbors for Each Domain
        """

        lookIDPad = np.pad(self.lookUpID, ( (1, 1), (1, 1), (1, 1)), 'constant', constant_values=-1)
        lookPerI = np.zeros_like(lookIDPad)
        lookPerJ = np.zeros_like(lookIDPad)
        lookPerK = np.zeros_like(lookIDPad)

        if (self.Domain.periodic[0] == True):
            lookIDPad[0,:,:]  = lookIDPad[-2,:,:]
            lookIDPad[-1,:,:] = lookIDPad[1,:,:]
            lookPerI[0,:,:] = 1
            lookPerI[-1,:,:] = -1

        if (self.Domain.periodic[1] == True):
            lookIDPad[:,0,:]  = lookIDPad[:,-2,:]
            lookIDPad[:,-1,:] = lookIDPad[:,1,:]
            lookPerJ[:,0,:] = 1
            lookPerJ[:,-1,:] = -1

        if (self.Domain.periodic[2] == True):
            lookIDPad[:,:,0]  = lookIDPad[:,:,-2]
            lookIDPad[:,:,-1] = lookIDPad[:,:,1]
            lookPerK[:,:,0] = 1
            lookPerK[:,:,-1] = -1

        cc = 0
        for f in self.Orientation.faces.values():
            cx = f['ID'][0]
            cy = f['ID'][1]
            cz = f['ID'][2]
            self.neighborF[cc]      = lookIDPad[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerF[cc,0] = lookPerI[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerF[cc,1] = lookPerJ[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerF[cc,2] = lookPerK[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            cc = cc + 1

        cc = 0
        for e in self.Orientation.edges.values():
            cx = e['ID'][0]
            cy = e['ID'][1]
            cz = e['ID'][2]
            self.neighborE[cc]      = lookIDPad[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerE[cc,0] = lookPerI[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerE[cc,1] = lookPerJ[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerE[cc,2] = lookPerK[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            cc = cc + 1


        cc = 0
        for c in self.Orientation.corners.values():
            cx = c['ID'][0]
            cy = c['ID'][1]
            cz = c['ID'][2]
            self.neighborC[cc]      = lookIDPad[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerC[cc,0] = lookPerI[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerC[cc,1] = lookPerJ[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerC[cc,2] = lookPerK[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            cc = cc + 1

        self.lookUpID = lookIDPad

    def loadBalanceStats(self):
        own = self.ownNodes
        ownGrid =  self.grid[own[0][0]:own[0][1],
                             own[1][0]:own[1][1],
                             own[2][0]:own[2][1]]
        self.poreNodes = np.sum(ownGrid)

def genDomainSubDomain(rank,size,subDomains,nodes,periodic,domainFile,dataRead):


    numSubDomains = np.prod(subDomains)
    if rank == 0:
        if (size != numSubDomains):
            print("Number of Subdomains Must Equal Number of Processors!")
            #exit()

    totalNodes = np.prod(nodes)

    ### Get Domain INFO for All Procs ###
    domainSize,sphereData = dataRead(domainFile)
    domain = Domain(nodes = nodes, domainSize = domainSize, subDomains = subDomains, periodic = periodic)
    domain.getdXYZ()
    domain.getSubNodes()


    orient = Orientation()

    sD = subDomain(Domain = domain, ID = rank, subDomains = subDomains, Orientation = orient) ## Switch to PROC ID?
    sD.getInfo()
    sD.getNeighbors()
    sD.getXYZ()
    sD.genDomain(sphereData)

    loadBalancing = True
    if loadBalancing:
        sD.loadBalanceStats()
        loadData = [sD.ID,sD.poreNodes,sD.ownNodesTotal]
        loadData = comm.gather(loadData, root=0)
        if rank == 0:
            sumPoreNodes = 0
            sumTotalNodes = 0
            for ld in loadData:
                sumPoreNodes = sumPoreNodes + ld[1]
                sumTotalNodes = sumTotalNodes + ld[2]
            print("Total Nodes",sumTotalNodes,"Pore Nodes",sumPoreNodes)
            print("Ideal Load Balancing is %2.1f%%" %(1./numSubDomains*100.))
            for ld in loadData:
                p = ld[1]/sumPoreNodes*100.
                t = ld[2]/sumTotalNodes*100.
                print("Rank: %i has %2.1f%% of the Pore Nodes and %2.1f%% of the total Nodes" %(ld[0],p,t))


    return domain,sD
