import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 3:
    print(
        "Possible usage: python3.11 combine_matlabfss.py <resultsFolder> <dataset>"
    )
    sys.exit(1)
else:        
    resultsFolder = Path(sys.argv[1])
    dataset = sys.argv[2]

# === === === ===
# Feature Scores
with open(resultsFolder.joinpath(f"{dataset}feature_scores.pkl"), "rb") as handle:
    pyFSS_featurescores = pickle.load(handle)

# Reading weights from IRelief
IRlf_folder = resultsFolder.joinpath("IRelief")
IRlf_featurescores_30 = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}WeightsI_30.csv"), header=None
)
IRlf_featurescores_50 = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}WeightsI_50.csv"), header=None
)
IRlf_featurescores_70 = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}WeightsI_70.csv"), header=None
)
IRlf = pd.Series(
    data=np.zeros(
        IRlf_featurescores_30.shape[0]*IRlf_featurescores_30.shape[1]*3
    ), name="IRlf"
)
i = 0
for IRlf_featurescores in [
        IRlf_featurescores_30, IRlf_featurescores_50, IRlf_featurescores_70
]:
    for iteration in IRlf_featurescores.columns:
        for feature in IRlf_featurescores.index:
            IRlf[i] = IRlf_featurescores.at[feature, iteration]
            i += 1
pyFSS_featurescores["IRlf"] = IRlf

# Reading weights from LHRelief
LHRlf_folder = resultsFolder.joinpath("LHRelief")
LHRlf_featurescores_30 = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}WeightsLM_30.csv"), header=None
)
LHRlf_featurescores_50 = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}WeightsLM_50.csv"), header=None
)
LHRlf_featurescores_70 = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}WeightsLM_70.csv"), header=None
)
LHRlf = pd.Series(
    data=np.zeros(
        LHRlf_featurescores_30.shape[0]*LHRlf_featurescores_30.shape[1]*3
    ), name="LHRlf"
)
i = 0
for LHRlf_featurescores in [
        LHRlf_featurescores_30, LHRlf_featurescores_50, LHRlf_featurescores_70
]:
    for iteration in LHRlf_featurescores.columns:
        for feature in LHRlf_featurescores.index:
            LHRlf[i] = LHRlf_featurescores.at[feature, iteration]
            i += 1
pyFSS_featurescores["LHRlf"] = LHRlf


# === === === ===
# Ranks
with open(resultsFolder.joinpath(f"{dataset}ranks.pkl"), "rb") as handle:
    pyFSS_ranks = pickle.load(handle)

def get_rankings(_importances):
    n_features = len(_importances)

    # Sorted features from most important to least
    sortedfeatures = sorted(
        range(n_features), key=lambda i: _importances[i], reverse=True
    )

    # Rank of each feature
    feature_ranks = np.zeros(n_features)
    for r, f in enumerate(sortedfeatures):
        feature_ranks[f] = r

    return feature_ranks

# Top 10 Features of IRelief
IRlf_ranks_30 = IRlf_featurescores_30.apply(get_rankings)
IRlf_ranks_50 = IRlf_featurescores_50.apply(get_rankings)
IRlf_ranks_70 = IRlf_featurescores_70.apply(get_rankings)

# Top 10 features x 50 iterations x 3 number of observations
IRlf_ranks = pd.Series(data=np.zeros(10*50*3), name="IRlf")
i = 0
for IRlf_ranks_df in [IRlf_ranks_30, IRlf_ranks_50, IRlf_ranks_70]:
    for itr in range(50):
        for rank in range(10):
            top_feature = IRlf_ranks_df.loc[:,itr][
                IRlf_ranks_df.loc[:,itr] == rank
            ].index[0]

            IRlf_ranks[i] = top_feature
            i += 1
pyFSS_ranks["IRlf"] = IRlf_ranks

# Top 10 Features of LHRelief
LHRlf_ranks_30 = LHRlf_featurescores_30.apply(get_rankings)
LHRlf_ranks_50 = LHRlf_featurescores_50.apply(get_rankings)
LHRlf_ranks_70 = LHRlf_featurescores_70.apply(get_rankings)

# Top 10 features x 50 iterations x 3 number of observations
LHRlf_ranks = pd.Series(data=np.zeros(10*50*3), name="LHRlf")
i = 0
for LHRlf_ranks_df in [LHRlf_ranks_30, LHRlf_ranks_50, LHRlf_ranks_70]:
    for itr in range(50):
        for rank in range(10):
            top_feature = LHRlf_ranks_df.loc[:,itr][
                LHRlf_ranks_df.loc[:,itr] == rank
            ].index[0]

            LHRlf_ranks[i] = top_feature
            i += 1
pyFSS_ranks["LHRlf"] = LHRlf_ranks


# === === === ===
# Elapsed Times
py_times = pd.read_csv(
    resultsFolder.joinpath(f"{dataset}elapsed_times.csv"), index_col=0
)
IRlf_times = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}_tI.csv"), header=None
)
LHRlf_times = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}_tLM.csv"), header=None
)
py_times["IRlf"]  = IRlf_times.stack().values
py_times["LHRlf"] = LHRlf_times.stack().values

pyFSS_featurescores.to_csv(f"{dataset}_featurescores.csv")
pyFSS_ranks.to_csv(f"{dataset}_ranks.csv")
py_times.to_csv(f"{dataset}_elapsedtimes.csv")

sys.exit(0)

# The deepest truth is the representation of the process by which truth itself is generated
