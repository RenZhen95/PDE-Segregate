import sys
import pickle
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from pdfSegregationBased_FS import segregateX_y
from pdfSegregationBased_FS import compute_OA, plot_overlapAreas

dsname = "geneExpressionCancerRNA"
with open("ranking_allFeaturesOA.pkl", "rb") as handle:
    _dict = pickle.load(handle)

with open("datasets/processedDatasets.pkl", "rb") as handle:
    dataset_dict = pickle.load(handle)

dataset = dataset_dict[dsname]
X = dataset['X']
y = dataset['y']
print(X.shape)
featureRanking = _dict[dsname]
topFeature = featureRanking[20]
lowFeature = featureRanking[-5]

y_segregatedGroup = segregateX_y(X, y)

# Intra-feature normalization and computing the overlap area among classes
yLabels = list(y_segregatedGroup.keys())
yLabels.sort()

# Plot
fig, ax = plt.subplots(2, 1, figsize=(5.5,9.))
topOA, topKernels, topLengths = compute_OA(yLabels, y_segregatedGroup, topFeature)
plot_overlapAreas(
    topKernels, topLengths, topOA, "Feature values (normalized)", "PDF", ax[0]
)
ax[0].set_title("Example of a top feature")
lowOA, lowKernels, lowLengths = compute_OA(yLabels, y_segregatedGroup, lowFeature)
plot_overlapAreas(
    lowKernels, lowLengths, lowOA, "Feature values (normalized)", "PDF", ax[1]
)
ax[1].set_title("Example of a feature of low importance")

plt.tight_layout()
plt.show()

sys.exit(0)
