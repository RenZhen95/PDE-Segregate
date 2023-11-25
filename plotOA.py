import pickle
import os, sys
import matplotlib.pyplot as plt

from pdfSegregationBased_FS import plot_overlapAreas, compute_OA

with open("top30Features.pkl", "rb") as handle:
    topFeatures = pickle.load(handle)

print(topFeatures)
sys.exit()
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6.75, 7))
compute_OA(yLabels, y_segregatedGroup, feat_idx)
plot_overlapAreas(_kernels, _lengths, computedOA, _featName, _title, _ax)

# Feature 1
plt.suptitle('Kernel Density Estimates')
fig.text(0.03, 0.5, r'Probability Density Function $f(x)$', va='center', rotation='vertical')

plt.subplots_adjust(right=0.825, top=0.915)
plt.show()
