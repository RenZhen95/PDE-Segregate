import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12)

# def _kernel(grid):
#     return np.exp(-1*(grid**2)/2)

from pdeseg import PDE_Segregate

# Implementation from scratch
X = np.array(
    [[1.0, 0.548, 0.823, 0.14, 0.0, 0.85, 0.62, 0.55]]
).T
y = np.array(
    [1, 1, 1, 2, 2, 3, 3, 3]
)

pdeSegregate = PDE_Segregate(k=2, lower_end=-1.5, upper_end=2.5)
pdeSegregate.fit(X, y)

fig, ax = plt.subplots(1, 1)
pdeSegregate.plot_overlapAreas(
    0, _combinations=(1, 2), legend=True, _ax=ax
)
ax.set_title("Test")
# ax.legend(bbox_to_anchor=(0.4, 1), loc='upper left', fontsize="large")
# ax.grid(visible=True, which="major", axis="both")
plt.tight_layout()
plt.show()
