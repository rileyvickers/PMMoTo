#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


import math
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
cnp.import_array()

cdef int numDirections = 26
_directions =  np.array([[-1,-1,-1,  0, 13],
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
              [ 0, 0, 1, 25, 24]],dtype=np.int8)

cdef cnp.int8_t [:,:] directions
directions = _directions



def _getDirection3D(self,
                    int ID,
                    list _localIndex,
                    int _availDirection,
                    cnp.ndarray[cnp.uint8_t, ndim=1] _direction,
                    cnp.ndarray[cnp.uint64_t, ndim=1] _node):



  cdef int i = _localIndex[0]
  cdef int j = _localIndex[1]
  cdef int k = _localIndex[2]

  cdef int d,ii,jj,kk,oppDir,returnCell,found

  if _availDirection > 0:
      d = 25
      found = 0
      while d > 0  and not found:
        if _direction[d] == 1:
          found = 1
        else:
          d = d - 1


      ii = directions[d][0]
      jj = directions[d][1]
      kk = directions[d][2]

      oppDir = directions[d][4]
      returnCell = _node[d]

      self.nodeInfo[returnCell].direction[oppDir] = 0
      self.nodeInfo[ID].direction[d] = 0
      self.nodeInfo[ID].availDirection -= 1
  else:
      returnCell = -1

  return returnCell


def  _genNodeDirections(self,
                        cnp.ndarray[cnp.uint8_t, ndim=3] _ind):

  cdef int i,j,k,c,d,ii,jj,kk,availDirection

  c = 0
  for fIndex in self.Orientation.faces:
      iL = self.subDomain.loopInfo[fIndex][0]
      jL = self.subDomain.loopInfo[fIndex][1]
      kL = self.subDomain.loopInfo[fIndex][2]
      for i in range(iL[0],iL[1],iL[2]):
          for j in range(jL[0],jL[1],jL[2]):
              for k in range(kL[0],kL[1],kL[2]):
                  if _ind[i+1,j+1,k+1] == 1:
                    availDirection = 0
                    for d in range(0,numDirections):
                        ii = directions[d][0] + 1
                        jj = directions[d][1] + 1
                        kk = directions[d][2] + 1
                        if (_ind[i+ii,j+jj,k+kk] == 1):
                            self.nodeInfo[c].direction[d] = 1
                            node = self.nodeTable[i+ii-1,j+jj-1,k+kk-1]
                            self.nodeInfo[c].nodeDirection[d] = node
                            availDirection += 1

                    self.nodeInfo[c].availDirection = availDirection
                    c = c + 1

  innerID = self.Orientation.numFaces
  iL = self.subDomain.loopInfo[innerID][0]
  jL = self.subDomain.loopInfo[innerID][1]
  kL = self.subDomain.loopInfo[innerID][2]
  for i in range(iL[0],iL[1],iL[2]):
      for j in range(jL[0],jL[1],jL[2]):
          for k in range(kL[0],kL[1],kL[2]):
            if _ind[i+1,j+1,k+1] == 1:
              availDirection = 0
              for d in range(0,numDirections):
                  ii = directions[d][0] + 1
                  jj = directions[d][1] + 1
                  kk = directions[d][2] + 1
                  if (_ind[i+ii,j+jj,k+kk] == 1):
                      self.nodeInfo[c].direction[d] = 1
                      node = self.nodeTable[i+ii-1,j+jj-1,k+kk-1]
                      self.nodeInfo[c].nodeDirection[d] = node
                      availDirection += 1

              self.nodeInfo[c].availDirection = availDirection
              c = c + 1

def  _genNodeInfo(self):
  pass

  # c = 0
  # for fIndex in self.Orientation.faces:
  #     iL = self.subDomain.loopInfo[fIndex][0]
  #     jL = self.subDomain.loopInfo[fIndex][1]
  #     kL = self.subDomain.loopInfo[fIndex][2]
  #     bID = list(self.Orientation.faces[fIndex]['ID'])
  #     for i in range(iL[0],iL[1],iL[2]):
  #         for j in range(jL[0],jL[1],jL[2]):
  #             for k in range(kL[0],kL[1],kL[2]):
  #                 if self.ind[i,j,k] == 1:
  #
  #                     iLoc = self.subDomain.indexStart[0]+i
  #                     jLoc = self.subDomain.indexStart[1]+j
  #                     kLoc = self.subDomain.indexStart[2]+k
  #
  #                     perFace  = self.subDomain.neighborPerF[fIndex]
  #
  #                     if perFace.any():
  #                         if iLoc >= self.Domain.nodes[0]:
  #                             iLoc = 0
  #                         elif iLoc < 0:
  #                             iLoc = self.Domain.nodes[0]-1
  #                         if jLoc >= self.Domain.nodes[1]:
  #                             jLoc = 0
  #                         elif jLoc < 0:
  #                             jLoc = self.Domain.nodes[1]-1
  #                         if kLoc >= self.Domain.nodes[2]:
  #                             kLoc = 0
  #                         elif kLoc < 0:
  #                             kLoc = self.Domain.nodes[2]-1
  #                     globIndex = iLoc*self.Domain.nodes[1]*self.Domain.nodes[2] +  jLoc*self.Domain.nodes[2] +  kLoc
  #
  #                     boundaryID = bID.copy()
  #                     if (i < 2):
  #                         boundaryID[0] = -1
  #                     elif (i >= self.ind.shape[0]-2):
  #                         boundaryID[0] = 1
  #                     if (j < 2):
  #                         boundaryID[1] = -1
  #                     elif (j >= self.ind.shape[1]-2):
  #                         boundaryID[1] = 1
  #                     if (k < 2):
  #                         boundaryID[2] = -1
  #                     elif(k >= self.ind.shape[2]-2):
  #                         boundaryID[2] = 1
  #
  #                     self.nodeInfo[c] = Node(ID=c,
  #                                             localIndex = [i,j,k],
  #                                             globalIndex = globIndex,
  #                                             boundary = True,
  #                                             boundaryID = boundaryID,
  #                                             inlet = self.subDomain.inlet[fIndex],
  #                                             outlet = self.subDomain.outlet[fIndex])
  #                     self.nodeTable[i,j,k] = c
  #                     c = c + 1
  #
  # innerID = self.Orientation.numFaces
  # iL = self.subDomain.loopInfo[innerID][0]
  # jL = self.subDomain.loopInfo[innerID][1]
  # kL = self.subDomain.loopInfo[innerID][2]
  # for i in range(iL[0],iL[1],iL[2]):
  #     for j in range(jL[0],jL[1],jL[2]):
  #         for k in range(kL[0],kL[1],kL[2]):
  #             if (self.ind[i,j,k] == 1):
  #                 globIndex = iLoc*self.Domain.nodes[1]*self.Domain.nodes[2] +  jLoc*self.Domain.nodes[2] +  kLoc
  #                 self.nodeInfo[c] = Node(ID = c,
  #                                         localIndex = [i,j,k],
  #                                         globalIndex = globIndex,
  #                                         boundary = False,
  #                                         boundaryID = [0,0,0],
  #                                         inlet = False,
  #                                         outlet = False)
  #                 self.nodeTable[i,j,k] = c
  #                 c = c + 1
