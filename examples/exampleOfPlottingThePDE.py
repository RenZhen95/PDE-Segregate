import pickle
import os, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))
from pde_segregate import PDE_Segregate

if len(sys.argv) < 3:
    print("Please enter appropriate file paths")
    sys.exit(0)
else:
    featureRankings_pkl = Path(sys.argv[1])
    processedDatasets = Path(sys.argv[2])

    with open(featureRankings_pkl, "rb") as handle:
        _dict = pickle.load(handle)
    
    with open(processedDatasets, "rb") as handle:
        dataset_dict = pickle.load(handle)

dsname = "lung"
dataset = dataset_dict[dsname]
X = dataset['X']
y = dataset['y']
print(X.shape)

# pdeSegregate = PDE_Segregate(X, y)
# with open(f"{dsname}_pdeSegregateObject.pkl", "wb") as handle:
#     pickle.dump(pdeSegregate, handle)

with open(f"{dsname}_pdeSegregateObject.pkl", "rb") as handle:
    pdeSegregate = pickle.load(handle)

featureRanking = _dict[dsname]
topFeatureIdx = featureRanking[20]
lowFeatureIdx = featureRanking[-10]

# Plot overlapping areas
fig, axs = plt.subplots(2, 1, figsize=(5.8, 6.5), sharex=True)
pdeSegregate.plot_overlapAreas(
    topFeatureIdx, feat_names=None, _ylim=None, _title="Class",
    show_samples=False, _ax=axs[0], legend=True
)
topOA = pdeSegregate.overlappingAreas[topFeatureIdx]
axs[0].set_title(f"Intersection area: {np.round(topOA, 3)}")
axs[0].grid(visible=True)

pdeSegregate.plot_overlapAreas(
    lowFeatureIdx, feat_names=None, _ylim=None, _title=None,
    show_samples=False, _ax=axs[1], legend=False
)
lowOA = pdeSegregate.overlappingAreas[lowFeatureIdx]
axs[1].set_title(f"Intersection area: {np.round(lowOA, 4)}")
axs[1].grid(visible=True)

plt.tight_layout()
plt.show()

sys.exit(0)
