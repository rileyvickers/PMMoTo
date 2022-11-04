"""
Algorithms for computing the skeleton of a binary image
"""
import numpy as np
from scipy import ndimage as ndi
from numbers import Integral

from ._skeletonize_3d_cy import _compute_thin_image

def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.
    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),) or (before, after)`` specifies
        a fixed start and end crop for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.
    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)

    if isinstance(crop_width, Integral):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], Integral):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(
                f'crop_width has an invalid length: {len(crop_width)}\n'
                f'crop_width should be a sequence of N pairs, '
                f'a single pair, or a single integer'
            )
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(
            f'crop_width has an invalid length: {len(crop_width)}\n'
            f'crop_width should be a sequence of N pairs, '
            f'a single pair, or a single integer'
        )

    slices = tuple(slice(a, ar.shape[i] - b)
                   for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


def skeletonize_3d(image):
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
    # make sure the image is 3D or 2D
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError("skeletonize_3d can only handle 2D or 3D images; "
                         f"got image.ndim = {image.ndim} instead.")
    image = np.ascontiguousarray(image)
    #image = img_as_ubyte(image, force_copy=False)

    # make an in image 3D and pad it w/ zeros to simplify dealing w/ boundaries
    # NB: careful here to not clobber the original *and* minimize copying
    image_o = np.copy(image)
    if image.ndim == 2:
        image_o = image[np.newaxis, ...]
    image_o = np.pad(image_o, pad_width=1, mode='constant')
    #
    # # #Inlet/Outlet
    image_o[0,:,:] = image_o[1,:,:]
    image_o[-1,:,:] = image_o[-2,:,:]
    # #
    # # ## Periodic
    image_o[:,0,:] = image_o[:,-2,:]
    image_o[:,-1,:] = image_o[:,1,:]
    image_o[:,:,0] = image_o[:,:,-2]
    image_o[:,:,-1] = image_o[:,:,1]

    # normalize to binary
    maxval = image_o.max()
    image_o[image_o != 0] = 1

    # do the computation
    image_o = np.asarray(_compute_thin_image(image_o))

    # crop it back and restore the original intensity range
    #image_o = crop(image_o, crop_width=1)
    if image.ndim == 2:
        image_o = image_o[0]
    image_o *= maxval

    return image_o
