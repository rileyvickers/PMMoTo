
import numpy as np
from mpi4py import MPI
from .. import communication
from ._skeletonize_3d_cy import _compute_thin_image
import math
comm = MPI.COMM_WORLD


class medialAxis(object):
    def __init__(self,Domain,subDomain):
        self.Domain = Domain
        self.subDomain = subDomain
        self.Orientation = subDomain.Orientation
        self.padding = np.zeros([3],dtype=np.int64)
        self.haloGrid = None
        self.MA = None


    def skeletonize_3d(self):
        """Compute the skeleton of a binary image.

        Thinning is used to reduce each connected component in a binary image
        to a single-pixel wide skeleton.

        Parameters
        ----------
        image : ndarray, 2D or 3D
            A binary image containing the objects to be skeletonized. Zeros
            represent background, nonzero values are foreground.

        Returns
        -------
        skeleton : ndarray
            The thinned image.

        See Also
        --------
        skeletonize, medial_axis

        Notes
        -----
        The method of [Lee94]_ uses an octree data structure to examine a 3x3x3
        neighborhood of a pixel. The algorithm proceeds by iteratively sweeping
        over the image, and removing pixels at each iteration until the image
        stops changing. Each iteration consists of two steps: first, a list of
        candidates for removal is assembled; then pixels from this list are
        rechecked sequentially, to better preserve connectivity of the image.

        The algorithm this function implements is different from the algorithms
        used by either `skeletonize` or `medial_axis`, thus for 2D images the
        results produced by this function are generally different.

        References
        ----------
        .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
               via 3-D medial surface/axis thinning algorithms.
               Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

        """
        self.haloGrid = np.ascontiguousarray(self.haloGrid)
        image_o = np.copy(self.haloGrid)

        # normalize to binary
        maxval = image_o.max()
        image_o[image_o != 0] = 1

        # do the computation
        image_o = np.asarray(_compute_thin_image(image_o))
        dim = image_o.shape
        self.MA = image_o#[self.halo[1]:dim[0]-self.halo[0],
                         # self.halo[3]:dim[1]-self.halo[2],
                         # self.halo[5]:dim[2]-self.halo[4]]

    def genPadding(self,grid):
        gridShape = self.Domain.subNodes
        factor = 0.5
        self.padding[0] = 40#math.ceil(gridShape[0]*factor)
        self.padding[1] = 40#math.ceil(gridShape[1]*factor)
        self.padding[2] = 40#math.ceil(gridShape[2]*factor)



def medialAxisEval(rank,size,Domain,subDomain,grid):
    sDMA = medialAxis(Domain = Domain,subDomain = subDomain)
    sDComm = communication.Comm(Domain = Domain,subDomain = subDomain,grid = grid)
    sDMA.genPadding(grid)
    sDMA.haloGrid,sDMA.halo = sDComm.haloCommunication(sDMA.padding)
    sDMA.skeletonize_3d()
    return sDMA
