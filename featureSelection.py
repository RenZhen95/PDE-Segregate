import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from mrmr import mrmr_classif
from skrebate import ReliefF, MultiSURF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from pdfSegregationBased_FS import get_overlappingAreasofPDF

def intersection(l1, l2):
    l3 = [value for value in l1 if value in l2]
    return l3

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

if len(sys.argv) < 4:
    print(
        "Possible usage: python3 featureSelection.py <processedDatasets> " +
        "<nRetainedFeatures> <fsResults_matlab>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])
    nRetainedFeatures = int(sys.argv[2])
    fsResults_matlab_folder = Path(sys.argv[3])

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)


# === === === ===
# Reading the feature rankings, output from the feature selection methods ran on MATLAB
ReliefI_dict = defaultdict()
ReliefLM_dict = defaultdict()
for f in os.scandir(fsResults_matlab_folder):
    ds_name = f.name.split('_')[0]
    if "ReliefI.csv" in f.name:
        tmpDF_RI = pd.read_csv(f, header=None).values
        ReliefI_dict[ds_name] = tmpDF_RI.reshape((tmpDF_RI.shape[0],))
    elif "ReliefLM" in f.name:
        # Taking only NN of 7 due to computational load
        # (suggested by author as good rule of thumb)
        tmpDF_RLM = pd.read_csv(f, header=None)[1].values
        ReliefLM_dict[ds_name] = tmpDF_RLM.reshape((tmpDF_RLM.shape[0],))


# === === === ===
# Carrying out feature selection for each dataset
dataset_inds_topFeatures = defaultdict()

inds_OAallFeatures = defaultdict()

for dataset in datasets_dict.keys():
    print(f"Dealing with {dataset} ... ")
    X = datasets_dict[dataset]['X']
    y = datasets_dict[dataset]['y']
    y_mapper = datasets_dict[dataset]['y_mapper']

    # From mrmr (https://github.com/smazzanti/mrmr)
    # Minimum redundancy - maximum relevance
    inds_topFeatures_mRMR = mrmr_classif(
        X=pd.DataFrame(X), y=pd.Series(y), K=nRetainedFeatures
    )

    # From scikit-rebate (https://github.com/EpistasisLab/scikit-rebate)
    # ReliefF
    RlfF = ReliefF(n_neighbors=7) # From Cai, 2014
    RlfF.fit(X, y)
    # MultiSURF
    MSurf = MultiSURF()
    MSurf.fit(X,y)

    # From scikit-learn
    # Mutual Information
    resMI = mutual_info_classif(X, y, n_neighbors=7, random_state=0)

    # Random forest ensemble data mining to increase information gain/reduce impurity
    rfGini = RandomForestClassifier(n_estimators=1000, criterion="gini", random_state=0)
    rfGini.fit(X,y)
    rfGini_featureImportance = rfGini.feature_importances_

    rfEntropy = RandomForestClassifier(n_estimators=1000, criterion="entropy", random_state=0)
    rfEntropy.fit(X,y)
    rfEntropy_featureImportance = rfEntropy.feature_importances_

    # Proposed algorithm
    # Overlapping Areas of PDFs
    resOA = get_overlappingAreasofPDF(X, y)
    # Reminder: The larger the overlapping areas of the PDFs, the weaker the
    # discriminating power of that feature
    resOA *= -1

    inds_topFeatures_RlfF   = get_indsTopnFeatures(RlfF.feature_importances_, nRetainedFeatures)
    inds_topFeatures_MSurf  = get_indsTopnFeatures(MSurf.feature_importances_, nRetainedFeatures)
    inds_topFeatures_RlfI   = get_indsTopnFeatures(ReliefI_dict[dataset], nRetainedFeatures)
    inds_topFeatures_RlfLM  = get_indsTopnFeatures(ReliefLM_dict[dataset], nRetainedFeatures)
    inds_topFeatures_MI     = get_indsTopnFeatures(resMI, nRetainedFeatures)
    inds_topFeatures_RFGini = get_indsTopnFeatures(rfGini_featureImportance, nRetainedFeatures)
    inds_topFeatures_RFEtry = get_indsTopnFeatures(rfEntropy_featureImportance, nRetainedFeatures)
    inds_topFeatures_OA     = get_indsTopnFeatures(resOA, nRetainedFeatures)
    inds_allFeatures_OA     = get_indsTopnFeatures(resOA, X.shape[1])

    inds_topFeatures = {
        "RlfF": inds_topFeatures_RlfF, "MSurf": inds_topFeatures_MSurf,
        "RlfI": inds_topFeatures_RlfI, "RlfLM": inds_topFeatures_RlfLM,
        "MI": inds_topFeatures_MI,
        "RFGini": inds_topFeatures_RFGini, "RFEtry": inds_topFeatures_RFEtry,
        "OA": inds_topFeatures_OA
    }

    dataset_inds_topFeatures[dataset] = inds_topFeatures
    inds_OAallFeatures[dataset] = inds_allFeatures_OA

with open(f"top{nRetainedFeatures}Features.pkl", "wb") as handle:
    pickle.dump(dataset_inds_topFeatures, handle)

with open(f"ranking_allFeaturesOA.pkl", "wb") as handle:
    pickle.dump(inds_OAallFeatures, handle)

sys.exit(0)
