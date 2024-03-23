# import gi; gi.require_version("Gtk", "3.0")
# from gi.repository import Gtk
# Gtk.init_check()

import numpy as np
np.random.seed(12)

def _kernel(grid):
    return np.exp(-1*(grid**2)/2)

# Implementation from scratch
Xgrid = np.linspace(0, 1, 500)
sN = np.array([0.548, 0.823, 1.0])
sP = np.array([0.0, 0.14])


def own_kde(_sample):
    scottFactor = len(_sample)**(-1/5)
    bw = scottFactor * _sample.std(ddof=1)
    pde_manual = 0.
    pde_kernels = []
    for s in _sample:
        _in = (s - Xgrid) / bw
        pde_manual = pde_manual + _kernel(_in)
        pde_kernels.append(_kernel(_in) / (len(_sample)*bw*np.sqrt(2*np.pi)))

    pde_manual = pde_manual / (len(_sample)*bw*np.sqrt(2*np.pi))

    return pde_manual, pde_kernels

pde_ownN, pde_kernelsN = own_kde(sN)
pde_ownP, pde_kernelsP = own_kde(sP)

# Scipy
from scipy.stats import gaussian_kde
pde_scipyN = np.reshape(gaussian_kde(sN)(Xgrid).T, Xgrid.shape)
pde_scipyP = np.reshape(gaussian_kde(sP)(Xgrid).T, Xgrid.shape)

# # sklearn
# from sklearn.neighbors import KernelDensity
# k_sk = KernelDensity(bandwidth=bw)
# k_sk.fit(s1[:, np.newaxis])
# log_pdf = k_sk.score_samples(Xgrid[:, np.newaxis])
# pde_sk = np.exp(log_pdf)

# Plots
import matplotlib.pyplot as plt

fig2, ax2 = plt.subplots(1, 1)
ax2.plot(Xgrid, pde_ownN, label=r"$\hat{p}_{\bar{f}_{2-}}$")
ax2.plot(Xgrid, pde_ownP, label=r"$\hat{p}_{\bar{f}_{2+}}$")

# # Plotting the individual kernels
# for k in pde_kernelsN:
#     ax2.plot(
#         Xgrid, k, linestyle="dotted",
#         color=np.array([76, 114, 176])/255
#     )
# for k in pde_kernelsP:
#     ax2.plot(
#         Xgrid, k, linestyle="dotted",
#         color=np.array([221, 132, 82])/255
#     )

# Getting intersection area
yStack = []
yStack.append(pde_ownN)
yStack.append(pde_ownP)
yIntersection = np.amin(yStack, axis=0)
OA = np.trapz(yIntersection, Xgrid)
_label = r"$A_{2} = $"
_label += str(round(OA,3))
fill_poly = ax2.fill_between(
    Xgrid, 0, yIntersection, label=_label,
    color="lightgray", edgecolor="lavender"
)
fill_poly.set_hatch('xxx')

ax2.grid(visible=True)
plt.xlim((-0.005, 1.005))
plt.xlabel(r"$\bar{f}_{2}$", fontsize='x-large')
plt.ylabel(r"$\hat{p}_{\bar{f}_{2}}$    ", fontsize='x-large', rotation=0)
plt.legend(fontsize='xx-large')
plt.tight_layout()
plt.show()
# fig2.savefig("gamm_step4.png", format="png")
