import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

from skrebate import ReliefF, MultiSURF
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

# ReliefF
RlfF = ReliefF(n_neighbors=9) # From Cai, 2014
RlfF.fit(X, y)
# MultiSURF
MSurf = MultiSURF()
MSurf.fit(X,y)
# Mutual Information
resMI = mutual_info_classif(X, y, n_neighbors=9, random_state=0)
# Overlapping Areas of PDFs
resOA = get_overlappingAreasofPDF(X, y)
# Reminder: The larger the overlapping areas of the PDFs, the weaker the
# discriminating power of that feature
resOA *= -1

inds_topFeatures_RlfF  = get_indsTopnFeatures(RlfF.feature_importances_)
inds_topFeatures_MSurf = get_indsTopnFeatures(MSurf.feature_importances_)
inds_topFeatures_MI    = get_indsTopnFeatures(resMI)
inds_topFeatures_OA    = get_indsTopnFeatures(resOA)

print(inds_topFeatures_RlfF)
print(inds_topFeatures_MSurf)
print(inds_topFeatures_MI)
print(inds_topFeatures_OA)

def intersection(l1, l2):
    l3 = [value for value in l1 if value in l2]
    return l3

iRlf_MI = intersection(inds_topFeatures_RlfF, inds_topFeatures_MI)
iRlf_MSurf = intersection(inds_topFeatures_RlfF, inds_topFeatures_MSurf)
iRlf_OA = intersection(inds_topFeatures_RlfF, inds_topFeatures_OA)
print(iRlf_MI)
print(len(iRlf_MI))
print(iRlf_MSurf)
print(len(iRlf_MSurf))
print(iRlf_OA)
print(len(iRlf_OA))

sys.exit(0)
