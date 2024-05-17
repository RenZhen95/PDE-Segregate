import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
)
from pde_segregate import PDE_Segregate

if len(sys.argv) < 4:
    print(
        "Possible usage: python3 featureSelection.py <processedDatasets> " +
        "<nRetainedFeatures> <savefolder>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])
    nRetainedFeatures = int(sys.argv[2])
    savefolder = Path(sys.argv[3])

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)

# === === === ===
# Carrying out feature selection for each dataset
dataset_inds_topFeatures = defaultdict()

elapsed_times_perDS = defaultdict()

for dataset in datasets_dict.keys():
    elapsed_times = defaultdict()

    print(f"Dealing with {dataset} ... ")
    X = datasets_dict[dataset]['X']
    y = datasets_dict[dataset]['y']
    y_mapper = datasets_dict[dataset]['y_mapper']

    # Proposed algorithm
    tPDE_start = process_time()
    pdeSegregate = PDE_Segregate(
        integration_method="trapz", delta=500, bw_method="scott",
        n=2, n_jobs=-1, mode="release", lower_end=-1.5, upper_end=2.5,
        averaging_method="weighted"
    )
    pdeSegregate.fit(X, y)
    tPDE_stop = process_time()
    tPDE = tPDE_stop - tPDE_start

    # === === === === === === ===
    # GETTING TOP N FEATURES
    inds_topFeatures_PDES = pdeSegregate.get_topnFeatures(
        nRetainedFeatures
    )
    dataset_inds_topFeatures[dataset] = inds_topFeatures_PDES

    # === === === === === === ===
    # GET ELAPSED TIME
    elapsed_times_perDS[dataset] = tPDE

with open(savefolder.joinpath(f"top{nRetainedFeatures}Features_PDE-S.pkl"), "wb") as handle:
    pickle.dump(dataset_inds_topFeatures, handle)
with open(savefolder.joinpath(f"fsElapsedTimes_PDE-S.pkl"), "wb") as handle:
    pickle.dump(elapsed_times_perDS, handle)

sys.exit(0)
