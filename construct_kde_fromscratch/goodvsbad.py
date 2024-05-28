import numpy as np
np.random.seed(12)

def _kernel(grid):
    return np.exp(-1*(grid**2)/2)

# Implementation from scratch
Xgrid = np.linspace(0, 1, 500)
# Feature 1
sN1 = np.array([1.0, 0.548, 0.823])
sP1 = np.array([0.14, 0.0])
# Feature 2
sN2 = np.array([1.0, 0.548, 0.823])
sP2 = np.array([0.65, 0.5, 0.75])

def own_kde(_sample):
    scottFactor = len(_sample)**(-1/5)
    bw = scottFactor * _sample.std(ddof=1)
    pde_manual = 0.
    for s in _sample:
        _in = (s - Xgrid) / bw
        pde_manual = pde_manual + _kernel(_in)
    pde_manual = pde_manual / (len(_sample)*bw*np.sqrt(2*np.pi))

    return pde_manual

pde_ownN1 = own_kde(sN1)
pde_ownP1 = own_kde(sP1)

pde_ownN2 = own_kde(sN2)
pde_ownP2 = own_kde(sP2)

# Plots
import matplotlib.pyplot as plt
plotlength = 9.0
fig2, ax2 = plt.subplots(
    2, 1, sharex=True, sharey=True, figsize=(plotlength/2, plotlength/1.33)
)

# Feature 1
ax2[0].plot(Xgrid, pde_ownN1, label="Negative")
ax2[0].plot(Xgrid, pde_ownP1, label="Positive")
# Getting intersection area
yStack1 = []
yStack1.append(pde_ownN1)
yStack1.append(pde_ownP1)
yIntersection1 = np.amin(yStack1, axis=0)
OA1 = np.trapz(yIntersection1, Xgrid)
fill_poly1 = ax2[0].fill_between(
    Xgrid, 0, yIntersection1,
    color="lightgray", edgecolor="lavender"
)
fill_poly1.set_hatch('xxx')

# # Samples
# ax2[0].vlines(
#     sN1, 0.0, 0.25,
#     color=np.array([76, 114, 176])/255, linewidth=2.5
# )
# ax2[0].vlines(
#     sP1, 0.0, 0.25,
#     color=np.array([221, 132, 82])/255, linewidth=2.5
# )

# Feature 2
ax2[1].plot(Xgrid, pde_ownN2, label="Negative")
ax2[1].plot(Xgrid, pde_ownP2, label="Positive")
# Getting intersection area
yStack2 = []
yStack2.append(pde_ownN2)
yStack2.append(pde_ownP2)
yIntersection2 = np.amin(yStack2, axis=0)
OA2 = np.trapz(yIntersection2, Xgrid)
fill_poly2 = ax2[1].fill_between(
    Xgrid, 0, yIntersection2,
    color="lightgray", edgecolor="lavender"
)
fill_poly2.set_hatch('xxx')

# # Samples
# ax2[1].vlines(
#     sN2, 0.0, 0.25,
#     color=np.array([76, 114, 176])/255, linewidth=2.5
# )
# ax2[1].vlines(
#     sP2, 0.0, 0.25,
#     color=np.array([221, 132, 82])/255, linewidth=2.5
# )

# x-ticks
plt.xlim((-0.005, 1.005))
plt.xlabel(r"$\boldsymbol{F'}$", fontsize='large')

# grids
ax2[0].grid(visible=True, which="major", axis="both")
ax2[1].grid(visible=True, which="major", axis="both")

# y-label
ax2[0].set_ylabel(r"$\hat{p}$", rotation=0, y=0.9, fontsize="large", labelpad=9.0)
ax2[1].set_ylabel(r"$\hat{p}$", rotation=0, y=0.9, fontsize="large", labelpad=9.0)

# titles
ax2[0].set_title("Feature A", loc="left")
ax2[1].set_title("Feature B", loc="left")

plt.tight_layout()

plt.show()
