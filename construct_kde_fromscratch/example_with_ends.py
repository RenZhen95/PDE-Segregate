import sys
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(122807528840384100672342137672332424406)

from pdeseg import PDE_Segregate

X = np.array(
    [
        [
            1.00, 0.45, 0.823, 0.151,
            0.14, 0.00, 0.205, 0.382, 0.701
        ]
    ]
).T
y = np.array(
    (
        2, 2, 2, 2,
        1, 1, 1, 1, 1
    )
)

plotlength = 9.0
fig, axs = plt.subplots(
    2, 1, figsize=(plotlength/2, plotlength/1.33)
)

pdeSegregate = PDE_Segregate(k=2, lower_end=0.0, upper_end=1.0)
pdeSegregate.fit(X, y)
pdeSegregate.plot_overlapAreas(
    0, _combinations=None, legend="intersection", _ax=axs[0]
)
axs[0].grid(visible=True, which="major", axis="both")

pdeSegregate_p5 = PDE_Segregate(k=2, lower_end=-1.0, upper_end=2.0)
pdeSegregate_p5.fit(X, y)
pdeSegregate_p5.plot_overlapAreas(
    0, _combinations=None, legend="intersection", _ax=axs[1]
)
axs[1].grid(visible=True, which="major", axis="both")

# Legends
axs[0].legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize="medium")
axs[1].legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize="medium")

# x-ticks
axs[0].set_xticks(np.arange(0.0, 1.2, 0.2))
axs[1].set_xticks(np.arange(-1.0, 2.5, 0.5))

# x-label
axs[0].set_xlabel("")
axs[1].set_xlabel(r"$\boldsymbol{F'}$")

# y-label
axs[0].set_ylabel(r"$\hat{p}$", rotation=0, y=0.9, fontsize="large", labelpad=9.0)
axs[1].set_ylabel(r"$\hat{p}$", rotation=0, y=0.9, fontsize="large", labelpad=9.0)

# titles
axs[0].set_title(r"KDEs evaluated from $0.0 \leq t \leq 1.0$", loc="left")
axs[1].set_title(r"KDEs evaluated from $-1.0 \leq t \leq 2.0$", loc="left")

fig.tight_layout()

plt.show()
