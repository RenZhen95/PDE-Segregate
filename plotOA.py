import pickle
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from pdfSegregationBased_FS import segregateX_y
from pdfSegregationBased_FS import compute_OA, plot_overlapAreas

dsname = "lung"
with open("ranking_allFeaturesOA.pkl", "rb") as handle:
    _dict = pickle.load(handle)

with open("datasets/processedDatasets.pkl", "rb") as handle:
    dataset_dict = pickle.load(handle)

dataset = dataset_dict[dsname]
X = dataset['X']
y = dataset['y']

featureRanking = _dict[dsname]
topFeature = featureRanking[12]
lowFeature = featureRanking[-1]

# fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6.75, 7))
y_segregatedGroup = segregateX_y(X, y)

# Intra-feature normalization and computing the overlap area among classes
yLabels = list(y_segregatedGroup.keys())
yLabels.sort()

# Plot
fig, ax = plt.subplots(1,2)
topOA, topKernels, topLengths = compute_OA(yLabels, y_segregatedGroup, topFeature)
plot_overlapAreas(
    topKernels, topLengths, topOA, "Feature values (normalized)", "PDF", ax[0]
)
lowOA, lowKernels, lowLengths = compute_OA(yLabels, y_segregatedGroup, lowFeature)
plot_overlapAreas(
    lowKernels, lowLengths, lowOA, "Feature values (normalized)", "PDF", ax[1]
)

plt.tight_layout()
plt.show()
