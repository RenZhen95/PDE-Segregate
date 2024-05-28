import sys
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(122807528840384100672342137672332424406)

from pdeseg import PDE_Segregate

# Class 1
class1_cluster1 = rng.normal(loc=0.0, scale=0.15, size=1200)
class1_cluster2 = rng.normal(loc=0.7, scale=0.1, size=1800)
class1mm = np.concatenate((class1_cluster1, class1_cluster2))

class1mm_calcsd = np.std(class1mm, ddof=1)

# Class 2
class2mm = rng.normal(loc=1.2*class1mm_calcsd, scale=0.1, size=1000)

# Class 3
class3_cluster1 = rng.normal(loc=1.5*class1mm_calcsd, scale=0.1, size=800)
class3_cluster2 = rng.normal(loc=4.0*class1mm_calcsd, scale=0.1, size=2000)
class3mm = np.concatenate((class3_cluster1, class3_cluster2))

X = np.concatenate((class1mm, class2mm, class3mm))

y1 = np.repeat(1, 3000)
y2 = np.repeat(2, 1000)
y3 = np.repeat(3, 2800)
y = np.concatenate((y1, y2, y3))

X = X.reshape((len(X), 1))

# pdeSegregate = PDE_Segregate(k=2, lower_end=-1.5, upper_end=2.5)
pdeSegregate = PDE_Segregate(k=2, lower_end=0.0, upper_end=1.0)
pdeSegregate.fit(X, y)

# Lazy way to get EPS of horizontal legend box
# fig, ax = plt.subplots(1,1)
# pdeSegregate.plot_overlapAreas(
#     0, _combinations=(1, 2), legend="class", _ax=ax
# )
# ax.legend(ncols=3, bbox_to_anchor=(1, 0.8), loc="upper right")

plotlength = 9.0
fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=(plotlength, plotlength/1.33)
)
pdeSegregate.plot_overlapAreas(
    0, _combinations=(1, 2), legend="intersection", _ax=axs[0,0]
)
axs[0,0].grid(visible=True, which="major", axis="both")
pdeSegregate.plot_overlapAreas(
    0, _combinations=(1, 3), legend="intersection", _ax=axs[1,0]
)
axs[1,0].grid(visible=True, which="major", axis="both")
pdeSegregate.plot_overlapAreas(
    0, _combinations=(2, 3), legend="intersection", _ax=axs[0,1]
)
axs[0,1].grid(visible=True, which="major", axis="both")
pdeSegregate.plot_overlapAreas(
    0, _combinations=None, legend="intersection", _ax=axs[1,1]
)
axs[1,1].grid(visible=True, which="major", axis="both")

# Legends
axs[0,0].legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize="medium")
axs[0,1].legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize="medium")
axs[1,0].legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize="medium")
axs[1,1].legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize="medium")

# x-limit
axs[1,0].set_xlim((0.0, 1.0))
axs[1,1].set_xlim((0.0, 1.0))

# x-ticks
axs[1,0].set_xticks(np.arange(0.0, 1.2, 0.2))
axs[1,1].set_xticks(np.arange(0.0, 1.2, 0.2))                    

# x-label
axs[0,0].set_xlabel("")
axs[0,1].set_xlabel("")
axs[1,0].set_xlabel(r"$\boldsymbol{F'}$")
axs[1,1].set_xlabel(r"$\boldsymbol{F'}$")

# titles
axs[0,0].set_title(r"$k=2$ | Combination (1,2)", loc="left")
axs[0,1].set_title(r"$k=2$ | Combination (1,3)", loc="left")
axs[1,0].set_title(r"$k=2$ | Combination (2,3)", loc="left")
axs[1,1].set_title(r"$k=|\mathcal{C}|$", loc="left")

fig.tight_layout()

plt.show()
