import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

"""
For the Electrical datasets, 3 % of 4000 would be roughly 120
"""

if len(sys.argv) < 4:
    print(
        "Possible usage: python3.11 evaluate_fss.py <resultsFolder> " +
        "<trueSignatures> <nTop>"
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])
    trueSignatures_folder = Path(sys.argv[2])
    nTop = int(sys.argv[3])

# Reading the feature scores
feature_scores_df = pd.read_csv(
    resultsFolder.joinpath("SDIfeaturescores.csv"), index_col=0
)

# Reading the top 10 features
ranks_df = pd.read_csv(
    resultsFolder.joinpath("SDIranks.csv"), index_col=0
)
# Narrow threshold if desired
only_retain = np.arange(0, nTop, 1)
ranks_df = ranks_df[ranks_df["rank"].isin(only_retain)]
print(ranks_df)

# Reading the elapsed times for each FSS method
elapsed_times_df = pd.read_csv(
    resultsFolder.joinpath("SDIelapsedtimes.csv"), index_col=0
)

# Getting true signatures
nClass2_idxs = [16, 43, 70, 97]
nClass3_idxs = [17, 44, 71, 98]
nClass4_idxs = [18, 45, 72, 99]

# IMPORTANT NOTE: R indexing starts from 1 just like MATLAB !!
def get_trueSignatures(_idxs):
    _trueSignatures_dict = defaultdict()
    for _i, _it in enumerate(_idxs):
        _true_signatures = pd.read_csv(
            trueSignatures_folder.joinpath(f"{_it}_trueSignatures.csv"), header=None
        )
        _trueSignatures_dict[_i] = _true_signatures - 1
    return _trueSignatures_dict

nClass2_trueSignatures = get_trueSignatures(nClass2_idxs)
nClass3_trueSignatures = get_trueSignatures(nClass3_idxs)
nClass4_trueSignatures = get_trueSignatures(nClass4_idxs)
trueSignatures = {
    2.0: nClass2_trueSignatures,
    3.0: nClass3_trueSignatures,
    4.0: nClass4_trueSignatures,
}

# Features Summary
features = feature_scores_df["feature"].to_numpy()
nFeatures = len(list(set(features)))

# Parameters for success metric according to Canedo, 2012
# Total number of irrelevant features
It = 4000
# Total number of relevant features
Rt = 60
# Alpha
_alpha = min(0.5, Rt/It)

print(f"Total irrelevant features : {It}")
print(f"Total relevant features   : {Rt}")

def count_instances_intop120(_top120):
    list_FSS = list(_top120.columns)
    list_FSS.remove("rank")
    list_FSS.remove("iteration")
    list_FSS.remove("nClass")

    nClass = list(set(_top120["nClass"].to_numpy()))[0]
    itr = list(set(_top120["iteration"].to_numpy()))[0]

    _truegenes = np.reshape(trueSignatures[nClass][itr].values, -1)

    count = pd.Series(data=np.zeros(len(list_FSS)), index=list_FSS)

    for i in _top120.index:
        for fss in count.index:
            if _top120.at[i, fss] in _truegenes:
                count.at[fss] += 1

    return count

groupeddf_relvcount = ranks_df.groupby(["iteration", "nClass"]).apply(
    count_instances_intop120
)

def check_ifrelevantistop(_top120):
    list_FSS = list(_top120.columns)
    list_FSS.remove("rank")
    list_FSS.remove("iteration")
    list_FSS.remove("nClass")

    nClass = list(set(_top120["nClass"].to_numpy()))[0]
    itr = list(set(_top120["iteration"].to_numpy()))[0]

    _truegenes = np.reshape(trueSignatures[nClass][itr].values, -1)

    # 3 dimensions, 20 genes per dimension = 60 true genes
    _top120_60first = _top120.iloc[:60,:]

    relv_atthetop = pd.Series(
        data=[False for i in range(len(list_FSS))], index=list_FSS
    )

    for fss in list_FSS:
        if set(_top120_60first[fss].values) == set(_truegenes):
            relv_atthetop[fss] = True

    return relv_atthetop

groupeddf_relvattop = ranks_df.groupby(["iteration", "nClass"]).apply(
    check_ifrelevantistop
)

# Computing the success rate as defined by Canedo, 2012
fss_list = list(groupeddf_relvattop.columns)
average_successrate = pd.DataFrame(
    data=np.zeros((3, len(fss_list))), index=[2.0, 3.0, 4.0], columns=fss_list
)

def calculate_success(_x):
    """
    Calculating the success rate as defined by Canedo, 2012
    """
    success_col = _x / Rt
    IRatio = _alpha * ((nTop - _x) / It)

    return (success_col - IRatio) * 100

def calculate_success_per_nClass(nClass):
    relvcount = groupeddf_relvcount.xs(nClass, level=1)
    relvattop = groupeddf_relvattop.xs(nClass, level=1)

    groupeddf_success = relvcount.apply(calculate_success)

    for fss in groupeddf_success.columns:
        for it_ in groupeddf_success.index:
            if relvattop.at[it_, fss]:
                groupeddf_success.at[it_, fss] = 100.0

    return groupeddf_success.mean()

avrsuccess_nClass2 = calculate_success_per_nClass(2.0)
avrsuccess_nClass3 = calculate_success_per_nClass(3.0)
avrsuccess_nClass4 = calculate_success_per_nClass(4.0)

average_successrate.loc[2.0] = avrsuccess_nClass2
average_successrate.loc[3.0] = avrsuccess_nClass3
average_successrate.loc[4.0] = avrsuccess_nClass4

print(average_successrate)

average_successrate.to_csv(f"{nTop}_SDIsuccessrates.csv")

sys.exit(0)
