# import gi; gi.require_version("Gtk", "3.0")
# from gi.repository import Gtk
# Gtk.init_check()

import numpy as np
np.random.seed(12)

def _kernel(grid):
    return np.exp(-1*(grid**2)/2)

# Implementation from scratch
Xgrid = np.linspace(0, 1, 500)
s1 = np.array([1.0, 0.548, 0.823])
s2 = np.array([0.14, 0.0])
s3 = np.array([0.85, 0.62, 0.55])


def own_kde(_sample):
    scottFactor = len(_sample)**(-1/5)
    bw = scottFactor * _sample.std(ddof=1)
    pde_manual = 0.
    for s in _sample:
        _in = (s - Xgrid) / bw
        pde_manual = pde_manual + _kernel(_in)
    pde_manual = pde_manual / (len(_sample)*bw*np.sqrt(2*np.pi))

    return pde_manual

pde_own1 = own_kde(s1)
pde_own2 = own_kde(s2)
pde_own3 = own_kde(s3)

# # Scipy
# from scipy.stats import gaussian_kde
# pde_scipyN = np.reshape(gaussian_kde(sN)(Xgrid).T, Xgrid.shape)
# pde_scipyP = np.reshape(gaussian_kde(sP)(Xgrid).T, Xgrid.shape)

# # sklearn
# from sklearn.neighbors import KernelDensity
# k_sk = KernelDensity(bandwidth=bw)
# k_sk.fit(s1[:, np.newaxis])
# log_pdf = k_sk.score_samples(Xgrid[:, np.newaxis])
# pde_sk = np.exp(log_pdf)

# Plots
import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
# ax[0].plot(Xgrid, pde_ownN)
# ax[0].plot(Xgrid, pde_ownP)
# ax[0].set_title("Implementation from scratch")
# ax[1].plot(Xgrid, pde_scipyN)
# ax[1].plot(Xgrid, pde_scipyP)
# ax[1].set_title("scipy.stats.gaussian_kde")
# ax[0].grid(visible=True)
# ax[1].grid(visible=True)

# Getting intersection area
yStack = []

fig2, ax2 = plt.subplots(1, 1)
# ax2.plot(Xgrid, pde_own1, label=r"$\hat{p}_{\bar{f}_{2,c_1}}$")
# yStack.append(pde_own1)
ax2.plot(Xgrid, pde_own1, linestyle="dotted")


ax2.plot(Xgrid, pde_own2, label=r"$\hat{p}_{\bar{f}_{2,c_2}}$")
yStack.append(pde_own2)
# ax2.plot(Xgrid, pde_own2, linestyle="dotted")

ax2.plot(Xgrid, pde_own3, label=r"$\hat{p}_{\bar{f}_{2,c_3}}$")
yStack.append(pde_own3)
# ax2.plot(Xgrid, pde_own3, linestyle="dotted")

yIntersection = np.amin(yStack, axis=0)
OA = np.trapz(yIntersection, Xgrid)
_label = r"$A_{2,c_{2}c_{3}} = $"
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
plt.legend(fontsize='xx-large', loc="upper left", bbox_to_anchor=(0.47, 1.02))
plt.tight_layout()
plt.show()
