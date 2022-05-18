#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import math
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
cnp.import_array()


def _fixInterfaceCalc(self,tree,faceID):

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
                orderL[nM] = cM
                orderL[nN] = cN

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
