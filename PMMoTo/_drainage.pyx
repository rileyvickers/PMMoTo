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

# directions ={0 :{'ID':[-1,-1,-1],'index': 0 ,'oppIndex':  13},
#                       1 :{'ID':[-1,-1, 1],'index': 1 ,'oppIndex':  12},
#                       2 :{'ID':[-1,-1, 0],'index': 2 ,'oppIndex':  14},
#                       3 :{'ID':[-1, 1,-1],'index': 3 ,'oppIndex':  10},
#                       4 :{'ID':[-1, 1, 1],'index': 4 ,'oppIndex':   9},
#                       5 :{'ID':[-1, 1, 0],'index': 5 ,'oppIndex':  11},
#                       6 :{'ID':[-1, 0,-1],'index': 6 ,'oppIndex':  16},
#                       7 :{'ID':[-1, 0, 1],'index': 7 ,'oppIndex':  15},
#                       8 :{'ID':[-1, 0, 0],'index': 8 ,'oppIndex':  17},
#                       9 :{'ID':[ 1,-1,-1],'index': 9 ,'oppIndex':   4},
#                       10:{'ID':[ 1,-1, 1],'index': 10 ,'oppIndex':  3},
#                       11:{'ID':[ 1,-1, 0],'index': 11 ,'oppIndex':  5},
#                       12:{'ID':[ 1, 1,-1],'index': 12 ,'oppIndex':  1},
#                       13:{'ID':[ 1, 1, 1],'index': 13 ,'oppIndex':  0},
#                       14:{'ID':[ 1, 1, 0],'index': 14 ,'oppIndex':  2},
#                       15:{'ID':[ 1, 0,-1],'index': 15 ,'oppIndex':  7},
#                       16:{'ID':[ 1, 0, 1],'index': 16 ,'oppIndex':  6},
#                       17:{'ID':[ 1, 0, 0],'index': 17 ,'oppIndex':  8},
#                       18:{'ID':[ 0,-1,-1],'index': 18 ,'oppIndex': 22},
#                       19:{'ID':[ 0,-1, 1],'index': 19 ,'oppIndex': 21},
#                       20:{'ID':[ 0,-1, 0],'index': 20 ,'oppIndex': 23},
#                       21:{'ID':[ 0, 1,-1],'index': 21 ,'oppIndex': 19},
#                       22:{'ID':[ 0, 1, 1],'index': 22 ,'oppIndex': 18},
#                       23:{'ID':[ 0, 1, 0],'index': 23 ,'oppIndex': 20},
#                       24:{'ID':[ 0, 0,-1],'index': 24 ,'oppIndex': 25},
#                       25:{'ID':[ 0, 0, 1],'index': 25 ,'oppIndex': 24},
#                      }


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

# def _validDirection(self,c):
#       self.direction[c] = 1
#
# def _setNodeDirection(self,c,node):
#       self.nodeDirection[c] = node


def  _genNodeDirections(self,
                        cnp.ndarray[cnp.uint8_t, ndim=3] _ind):

  cdef int i,j,k,c,d,ii,jj,kk,availDirection

  cdef int NX = _ind.shape[0]
  cdef int NY = _ind.shape[1]
  cdef int NZ = _ind.shape[2]

  c = 0
  for i in range(1,NX-1):
      for j in range(1,NY-1):
          for k in range(1,NZ-1):
              if (_ind[i,j,k] == 1):
                  availDirection = 0
                  for d in range(0,numDirections):
                      ii = directions[d][0]
                      jj = directions[d][1]
                      kk = directions[d][2]
                      if (_ind[i+ii,j+jj,k+kk] == 1):
                          #self.nodeInfo[c].validDirection(d)
                          self.nodeInfo[c].direction[d] = 1
                          node = self.nodeTable[i+ii-1,j+jj-1,k+kk-1]
                          #self.nodeInfo[c].setNodeDirection(d,node)
                          self.nodeInfo[c].nodeDirection[d] = node
                          availDirection += 1

                  self.nodeInfo[c].availDirection = availDirection
                  self.nodeInfo[c].saveDirection = self.nodeInfo[c].availDirection

                  c = c + 1
