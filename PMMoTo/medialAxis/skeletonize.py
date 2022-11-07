"""
Algorithms for computing the skeleton of a binary image
"""
import numpy as np
from scipy import ndimage as ndi
from numbers import Integral

from ._skeletonize_3d_cy import _compute_thin_image

def skeletonize_3d(image,perPad):
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

    image = np.ascontiguousarray(image)

    image_o = np.copy(image)

    #perPad = 30
    padDim = ((0, 0), (perPad, perPad), (perPad, perPad))
    image_o = np.pad(image_o, padDim, mode='constant')
    #
    # # #Inlet/Outlet
    #image_o[0,:,:] = image_o[1,:,:]
    #image_o[-1,:,:] = image_o[-2,:,:]
    # #
    # ## Periodic
    #image_o[0:10,:,:] = image_o[-12:-2,:,:]
    #image_o[-11:-1,:,:] = image_o[1:11,:,:]
    image_o[:,0:padDim[1][0],:] = image_o[:,-padDim[1][1]-padDim[1][0]-1:-padDim[1][0]-1,:]
    image_o[:,-padDim[1][1]-1:,:] = image_o[:,padDim[1][0]:padDim[1][0]+padDim[1][1]+1,:]
    image_o[:,:,0:padDim[2][0]] = image_o[:,:,-padDim[2][1]-padDim[2][0]-1:-padDim[2][0]-1]
    image_o[:,:,-padDim[1][1]-1:] = image_o[:,:,padDim[2][0]:padDim[2][0]+padDim[2][1]+1]

    # normalize to binary
    maxval = image_o.max()
    image_o[image_o != 0] = 1

    # do the computation
    image_o = np.asarray(_compute_thin_image(image_o))

    #image_o = image_o[:,padDim[1][0]:-padDim[1][1],padDim[2][0]:-padDim[2][1]]


    return image_o
