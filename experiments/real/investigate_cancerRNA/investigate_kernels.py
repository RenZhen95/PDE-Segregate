import pickle
import numpy as np
import matplotlib.pyplot as plt

# Old 0.0 - 1.0  : [16920, 12537, 14792]
# New -1.0 - 2.0 : [1842, 3525, 4368]
# MSurf          : [17547, 18492, 7949]

with open("pdes_new_kernels.pkl", "rb") as handle:
    newKernels = pickle.load(handle)

with open("pdes_old_kernels.pkl", "rb") as handle:
    oldKernels = pickle.load(handle)

XGrid_01 = np.linspace(0.0, 1.0, 1500)
XGrid_12 = np.linspace(-1.0, 2.0, 1500)

kernel1842 = oldKernels[1842]
kernel1842_c0 = kernel1842[0]
kernel1842_c1 = kernel1842[1]
kernel1842_c2 = kernel1842[2]
kernel1842_c3 = kernel1842[3]
kernel1842_c4 = kernel1842[4]
print(kernel1842_c0)
print(kernel1842_c1)
print(kernel1842_c2)
print(kernel1842_c3)
print(kernel1842_c4)

