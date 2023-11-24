import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

from mrmr import mrmr_classif
from skrebate import ReliefF, MultiSURF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from pdfSegregationBased_FS import get_overlappingAreasofPDF

# Taking the top nRetainedFeatures
def get_indsTopnFeatures(featImportances):
    res_tmp = -featImportances
    res_tmpSorted = np.sort(res_tmp)
    
    inds_topFeatures = []
    counter = 1
    for i in res_tmpSorted:
        inds_topFeatures.append(np.where(res_tmp==i)[0][0])
    
        counter += 1
        if counter==nRetainedFeatures:
            break

    return inds_topFeatures

if len(sys.argv) < 3:
    print(
        "Possible usage: python3 featureSelection.py <processedDatasets> " +
        "<nRetainedFeatures>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])
    nRetainedFeatures = int(sys.argv[2])

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)

dataset = datasets_dict["leuk"]
X = dataset['X']
y = dataset['y']
y_mapper = dataset['y_mapper']

# From mrmr (https://github.com/smazzanti/mrmr)
# Minimum redundancy - maximum relevance
inds_topFeatures_mRMR = mrmr_classif(
    X=pd.DataFrame(X), y=pd.Series(y), K=nRetainedFeatures
)

# From scikit-rebate (https://github.com/EpistasisLab/scikit-rebate)
# ReliefF
RlfF = ReliefF(n_neighbors=9) # From Cai, 2014
RlfF.fit(X, y)
# MultiSURF
MSurf = MultiSURF()
MSurf.fit(X,y)

# From scikit-learn
# Mutual Information
resMI = mutual_info_classif(X, y, n_neighbors=9, random_state=0)

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

inds_topFeatures_RlfF   = get_indsTopnFeatures(RlfF.feature_importances_)
inds_topFeatures_MSurf  = get_indsTopnFeatures(MSurf.feature_importances_)
inds_topFeatures_MI     = get_indsTopnFeatures(resMI)
inds_topFeatures_RFGini = get_indsTopnFeatures(rfGini_featureImportance)
inds_topFeatures_RFEtry = get_indsTopnFeatures(rfEntropy_featureImportance)
inds_topFeatures_OA     = get_indsTopnFeatures(resOA)

print(inds_topFeatures_RlfF)

def intersection(l1, l2):
    l3 = [value for value in l1 if value in l2]
    return l3

iRlf_MI = intersection(inds_topFeatures_RlfF, inds_topFeatures_MI)
iRlf_MSurf = intersection(inds_topFeatures_RlfF, inds_topFeatures_MSurf)
iRlf_OA = intersection(inds_topFeatures_RlfF, inds_topFeatures_OA)
iRlf_mRMR = intersection(inds_topFeatures_RlfF, inds_topFeatures_mRMR)
iRlf_RFGini = intersection(inds_topFeatures_RlfF, inds_topFeatures_RFGini)
iRlf_RFEtry = intersection(inds_topFeatures_RlfF, inds_topFeatures_RFEtry)

print(iRlf_MI)
print(len(iRlf_MI))
print(iRlf_MSurf)
print(len(iRlf_MSurf))
print(iRlf_OA)
print(len(iRlf_OA))
print(iRlf_mRMR)
print(len(iRlf_mRMR))
print(iRlf_RFGini)
print(len(iRlf_RFGini))
print(iRlf_RFEtry)
print(len(iRlf_RFEtry))

sys.exit(0)
