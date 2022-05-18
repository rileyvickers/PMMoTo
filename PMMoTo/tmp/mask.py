import numpy as np
from scipy import ndimage
a = np.zeros((11, 11))
a[5, 5] = 1
a[0, 1] = 1
struct1 = ndimage.generate_binary_structure(2, 1)
print(a)
ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)
