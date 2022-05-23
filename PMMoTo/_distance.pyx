#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


import math
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
cnp.import_array()


def _fixInterfaceCalc(self,
                     tree,
                     int faceID,
                     int lShape,
                     int dir,
                     cnp.ndarray[cnp.int32_t, ndim=2] _faceSolids,
                     cnp.ndarray[cnp.float32_t, ndim=3] _EDT,
                     cnp.ndarray[cnp.uint8_t, ndim=3] _visited,
                     double minD,
                     list coords,
                     cnp.ndarray[cnp.uint8_t, ndim=1] argOrder):


    cdef int i,l,m,n,endL,iShape
    cdef float maxD,d

    _orderG = np.ones((1,3), dtype=np.double)
    _orderL = np.ones((3), dtype=np.uint32)
    cdef cnp.uint32_t [:] orderL
    orderL = _orderL

    cdef int a0 = argOrder[0]
    cdef int a1 = argOrder[1]
    cdef int a2 = argOrder[2]

    cdef cnp.double_t [:] c0 = coords[a0]
    cdef cnp.double_t [:] c1 = coords[a1]
    cdef cnp.double_t [:] c2 = coords[a2]

    iShape = _faceSolids.shape[0]

    if (dir == 1):
        for i in range(0,iShape):

            if _faceSolids[i,argOrder[0]] < 0:
                endL = lShape
            else:
                endL = _faceSolids[i,argOrder[0]]

            distChanged = True
            l = 0
            while distChanged and l < endL:
                m = _faceSolids[i,argOrder[1]]
                n = _faceSolids[i,argOrder[2]]
                _orderG[0,a0] = c0[l]
                _orderG[0,a1] = c1[m]
                _orderG[0,a2] = c2[n]
                orderL[a0] = l
                orderL[a1] = m
                orderL[a2] = n

                maxD = _EDT[orderL[0],orderL[1],orderL[2]]
                if (maxD > minD):
                    d,ind = tree.query(_orderG,distance_upper_bound=maxD)
                    if d < maxD:
                        _EDT[orderL[0],orderL[1],orderL[2]] = d
                        distChanged = True
                        _visited[orderL[0],orderL[1],orderL[2]] = 1
                    elif _visited[orderL[0],orderL[1],orderL[2]] == 0:
                        distChanged = False
                l = l + 1

    if (dir == -1):
        for i in range(0,iShape):

            if _faceSolids[i,argOrder[0]] < 0:
                endL = 0
            else:
                endL = _faceSolids[i,argOrder[0]]

            distChanged = True
            l = lShape - 1

            while distChanged and l > endL:

                m = _faceSolids[i,argOrder[1]]
                n = _faceSolids[i,argOrder[2]]
                _orderG[0,a0] = c0[l]
                _orderG[0,a1] = c1[m]
                _orderG[0,a2] = c2[n]
                orderL[a0] = l
                orderL[a1] = m
                orderL[a2] = n

                maxD = _EDT[orderL[0],orderL[1],orderL[2]]
                if (maxD > minD):
                    d,ind = tree.query(_orderG,distance_upper_bound=maxD)
                    if d < maxD:
                        _EDT[orderL[0],orderL[1],orderL[2]] = d
                        distChanged = True
                        _visited[orderL[0],orderL[1],orderL[2]] = 1
                    elif _visited[orderL[0],orderL[1],orderL[2]] == 0:
                        distChanged = False
                l = l - 1
    return _EDT,_visited





def _getBoundarySolids(self,
                       int faceID,
                       int dir,
                       cnp.ndarray[cnp.uint8_t, ndim=1] argOrder,
                       int nS,
                       cnp.ndarray[cnp.uint8_t, ndim=3] _grid,
                       cnp.ndarray[cnp.int32_t, ndim=2] _solids):

    cdef int c,m,n

    _order = np.ones((3), dtype=np.uint32)
    cdef cnp.uint32_t [:] order
    order = _order

    cdef int a0 = argOrder[0]
    cdef int a1 = argOrder[1]
    cdef int a2 = argOrder[2]

    cdef int s0 = _grid.shape[a0]
    cdef int s1 = _grid.shape[a1]
    cdef int s2 = _grid.shape[a2]

    if (dir == 1):
        for m in range(0,s1):
            for n in range(0,s2):
                solid = False
                c = 0
                while not solid and c < s0:
                    order[a0] = c
                    order[a1] = m
                    order[a2] = n
                    if _grid[order[0],order[1],order[2]] == 0:
                        solid = True
                        _solids[nS,0:3] = order
                        _solids[nS,3] = faceID
                        nS = nS + 1
                    else:
                        c = c + 1
                if (not solid and c == s0):
                    order[a0] = -1
                    _solids[nS,0:3] = order
                    _solids[nS,3] = faceID
                    nS = nS + 1

    elif (dir == -1):
        for m in range(0,s1):
            for n in range(0,s2):
                solid = False
                c = s0 - 1
                while not solid and c > 0:
                    order[a0] = c
                    order[a1] = m
                    order[a2] = n
                    if _grid[order[0],order[1],order[2]] == 0:
                        solid = True
                        _solids[nS,0:3] = order
                        _solids[nS,3] = faceID
                        nS = nS + 1
                    else:
                        c = c - 1
                if (not solid and c == 0):
                    order[a0] = -1
                    _solids[nS,0:3] = order
                    _solids[nS,3] = faceID
                    nS = nS + 1
    return nS
