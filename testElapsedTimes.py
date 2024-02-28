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
        "Possible usage: python3 testElapsedTimes.py <processedDatasets>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)


# === === === ===
# Carrying out feature selection for each dataset
dataset_inds_topFeatures = defaultdict()

elapsed_times_perDS = defaultdict()

elapsed_times = defaultdict()

X = datasets_dict["colon"]['X']
y = datasets_dict["colon"]['y']
y_mapper = datasets_dict["colon"]['y_mapper']
print(X.shape)


# === === === === === === ===
# FEATURE RANKING METHODS
# From scikit-rebate (https://github.com/EpistasisLab/scikit-rebate)
# ReliefF
tRlfF_start = process_time()
RlfF = ReliefF(n_neighbors=7, n_jobs=-1) # From Cai, 2014
RlfF.fit(X, y)
tRlfF_stop = process_time()
tRlfF = tRlfF_stop - tRlfF_start

# MultiSURF
tMSurf_start = process_time()
MSurf = MultiSURF(n_jobs=-1)
MSurf.fit(X, y)
tMSurf_stop = process_time()
tMSurf = tMSurf_stop - tMSurf_start

# Proposed algorithm
# Overlapping Areas of PDEs
t_start = process_time()
pdeSegregate_single = PDE_Segregate(X, y, "trapz", delta=100, bw_method="scott", n_jobs=1)
t_stop = process_time()
tsingle = t_stop - t_start

t_start = process_time()
pdeSegregate_multi = PDE_Segregate(X, y, "trapz", delta=100, bw_method="scott", n_jobs=-1)
t_stop = process_time()
tmulti = t_stop - t_start

print("Time for single thread:", tsingle)
print("Intersection Areas    :", pdeSegregate_single.overlappingAreas[0:5])
print("Time for multi thread :", tmulti)
print("Intersection Areas    :", pdeSegregate_multi.overlappingAreas[0:5])

print("Time for RlfF         :", tRlfF)
print("Time for MultiSURF    :", tMSurf)

sys.exit(0)
