import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("pdes_new_kernels.pkl", "rb") as handle:
    newKernels = pickle.load(handle)

with open("pdes_old_kernels.pkl", "rb") as handle:
    oldKernels = pickle.load(handle)

with open("pdes_new_normalizedXdict.pkl", "rb") as handle:
    new_normalizedXdict = pickle.load(handle)

with open("pdes_old_normalizedXdict.pkl", "rb") as handle:
    old_normalizedXdict = pickle.load(handle)

XGrid_01 = np.linspace(0.0, 1.0, 1500)
XGrid_12 = np.linspace(-1.0, 2.0, 1500)

idx_0_btm = np.where((XGrid_12 > -0.0021))[0][0]
idx_0_top = np.where((XGrid_12 < 0.0021))[0][-1]

XGrid_12_0btm = XGrid_12[idx_0_btm]
XGrid_12_0top = XGrid_12[idx_0_top]

def compare_kernels_grid01(_fidx):
    oldkernel_f_c0 = oldKernels[_fidx][0]
    oldkernel_f_c1 = oldKernels[_fidx][1]
    oldkernel_f_c2 = oldKernels[_fidx][2]
    oldkernel_f_c3 = oldKernels[_fidx][3]
    oldkernel_f_c4 = oldKernels[_fidx][4]

    newkernel_f_c0 = newKernels[_fidx][0]
    newkernel_f_c1 = newKernels[_fidx][1]
    newkernel_f_c2 = newKernels[_fidx][2]
    newkernel_f_c3 = newKernels[_fidx][3]
    newkernel_f_c4 = newKernels[_fidx][4]

    print("Old kernels: Grid from 0 to 1")
    print(oldkernel_f_c0(XGrid_01))
    print(oldkernel_f_c1(XGrid_01))
    print(oldkernel_f_c2(XGrid_01))
    print(oldkernel_f_c3(XGrid_01))
    print(oldkernel_f_c4(XGrid_01))

    print("New kernels: Grid from 0 to 1")
    print(newkernel_f_c0(XGrid_01))
    print(newkernel_f_c1(XGrid_01))
    print(newkernel_f_c2(XGrid_01))
    print(newkernel_f_c3(XGrid_01))
    print(newkernel_f_c4(XGrid_01))

    c0_diff = oldkernel_f_c0(XGrid_01) - newkernel_f_c0(XGrid_01)
    c1_diff = oldkernel_f_c1(XGrid_01) - newkernel_f_c1(XGrid_01)
    c2_diff = oldkernel_f_c2(XGrid_01) - newkernel_f_c2(XGrid_01)
    c3_diff = oldkernel_f_c3(XGrid_01) - newkernel_f_c3(XGrid_01)
    c4_diff = oldkernel_f_c4(XGrid_01) - newkernel_f_c4(XGrid_01)

    print(f"Sum c0 diff: {c0_diff.sum()}")
    print(f"Sum c1 diff: {c1_diff.sum()}")
    print(f"Sum c2 diff: {c2_diff.sum()}")
    print(f"Sum c3 diff: {c3_diff.sum()}")
    print(f"Sum c4 diff: {c4_diff.sum()}")

def compare_kernels_grid12(_fidx):
    oldkernel_f_c0 = oldKernels[_fidx][0]
    oldkernel_f_c1 = oldKernels[_fidx][1]
    oldkernel_f_c2 = oldKernels[_fidx][2]
    oldkernel_f_c3 = oldKernels[_fidx][3]
    oldkernel_f_c4 = oldKernels[_fidx][4]

    newkernel_f_c0 = newKernels[_fidx][0]
    newkernel_f_c1 = newKernels[_fidx][1]
    newkernel_f_c2 = newKernels[_fidx][2]
    newkernel_f_c3 = newKernels[_fidx][3]
    newkernel_f_c4 = newKernels[_fidx][4]

    print("Old kernels: Grid from -1 to 2")
    print(oldkernel_f_c0(XGrid_12))
    print(oldkernel_f_c1(XGrid_12))
    print(oldkernel_f_c2(XGrid_12))
    print(oldkernel_f_c3(XGrid_12))
    print(oldkernel_f_c4(XGrid_12))

    print("New kernels: Grid from -1 to 2")
    print(newkernel_f_c0(XGrid_12))
    print(newkernel_f_c1(XGrid_12))
    print(newkernel_f_c2(XGrid_12))
    print(newkernel_f_c3(XGrid_12))
    print(newkernel_f_c4(XGrid_12))

    c0_diff = oldkernel_f_c0(XGrid_12) - newkernel_f_c0(XGrid_12)
    c1_diff = oldkernel_f_c1(XGrid_12) - newkernel_f_c1(XGrid_12)
    c2_diff = oldkernel_f_c2(XGrid_12) - newkernel_f_c2(XGrid_12)
    c3_diff = oldkernel_f_c3(XGrid_12) - newkernel_f_c3(XGrid_12)
    c4_diff = oldkernel_f_c4(XGrid_12) - newkernel_f_c4(XGrid_12)

    print(f"Sum c0 diff: {c0_diff.sum()}")
    print(f"Sum c1 diff: {c1_diff.sum()}")
    print(f"Sum c2 diff: {c2_diff.sum()}")
    print(f"Sum c3 diff: {c3_diff.sum()}")
    print(f"Sum c4 diff: {c4_diff.sum()}")

# Top features
# Old 0.0 - 1.0  : [16920, 12537, 14792]
# New -1.0 - 2.0 : [1842, 3525, 4368]
# MSurf          : [17547, 18492, 7949]
# MI             : [7949, 18127, 16855]

# compare_kernels_grid01(1842)
# compare_kernels_grid12(1842)

XGrid_12_0btm = XGrid_12[idx_0_btm]
XGrid_12_0top = XGrid_12[idx_0_top]
print(XGrid_12_0btm)
print(XGrid_12_0top)

print("Old kernels:")
print(oldKernels[1842][1]([0.0]))
print(oldKernels[1842][1]([-0.0015]))
print(oldKernels[1842][1]([0.0015]))

print("\nNew kernels:")
print(newKernels[1842][1]([0.0]))
print(newKernels[1842][1]([-0.0015]))
print(newKernels[1842][1]([0.0015]))

# print(old_normalizedXdict[1842][0])
# print(new_normalizedXdict[1842][0])
# print(old_normalizedXdict[1842][1])
# print(new_normalizedXdict[1842][1])
# print(old_normalizedXdict[1842][2])
# print(new_normalizedXdict[1842][2])
# print(old_normalizedXdict[1842][3])
# print(new_normalizedXdict[1842][3])
# print(old_normalizedXdict[1842][4])
# print(new_normalizedXdict[1842][4])


