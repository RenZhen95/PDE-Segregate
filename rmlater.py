import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

from anova import anova
from skrebate import ReliefF, MultiSURF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

from pde_segregate import PDE_Segregate

# Taking the top nRetainedFeatures
def get_indsTopnFeatures(featImportances, nCap):
    res_tmp = -featImportances
    res_tmpSorted = np.sort(res_tmp)

    inds_topFeatures = []
    counter = 1
    for i in res_tmpSorted:
        inds_topFeatures.append(np.where(res_tmp==i)[0][0])

        counter += 1
        if counter==nCap+1:
            break

    return inds_topFeatures

if len(sys.argv) < 2:
    print(
        "Possible usage: python3 featureSelection.py <processedDatasets>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)


# === === === ===
# Carrying out feature selection for each dataset
dataset_inds_topFeatures = defaultdict()

inds_OAallFeatures = defaultdict()

elapsed_times_perDS = defaultdict()

dataset = datasets_dict["geneExpressionCancerRNA"]

X = dataset['X']
X = X[:, 0:20]
y = dataset['y']
y_mapper = dataset['y_mapper']

# Proposed algorithm
# Overlapping Areas of PDEs
pdeSegregate = PDE_Segregate(
    X, y, "trapz", delta=100, bw_method="scott", pairwise=True, n_jobs=-1
)
pdeSegregate.fit()
print(pdeSegregate.get_scores())

sys.exit(0)
