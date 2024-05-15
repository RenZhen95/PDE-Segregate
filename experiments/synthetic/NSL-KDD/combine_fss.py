import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print(
        "Possible usage: python3.11 combine_fss.py <folder>"
    )
    sys.exit(1)
else:        
    folder = Path(sys.argv[1])

# === === === ===
# Feature Scores (other FS)
scores_df = pd.read_csv(folder.joinpath("OtherFS/scores_df.csv"), index_col=0)

# Reading weights from IRelief
IRlf_folder = folder.joinpath("IRelief")
IRlf_featurescores = pd.read_csv(
    IRlf_folder.joinpath(f"IReliefscores.csv"), header=None
)
scores_df["IRlf"] = IRlf_featurescores

# Reading weights from LHRelief
LHRlf_folder = folder.joinpath("LHRelief")
LHRlf_featurescores = pd.read_csv(
    LHRlf_folder.joinpath(f"LHReliefscores.csv"), header=None
)
scores_df["LHRlf"] = LHRlf_featurescores

# Reading scores from PDE-S
pdes_featurescores = pd.read_csv(
    folder.joinpath(f"PDE-S/PDE-S_scores_df.csv"), index_col=0
)
scores_df["PDE-S"] = pdes_featurescores["PDE-S"]


# === === === ===
# Ranks
ranks_df = pd.read_csv(folder.joinpath(f"OtherFS/ranks.csv"), index_col=0)

# Reading ranks from mRMR (Ding, 2005)
# Minus 1 because MATLAB indexing starts from 1 and not 0
# From MATLAB documentation:
#   If idx(3) is 5 :: The third most important featurey is the 10th column
mRMR_folder = folder.joinpath("mRMR")
mRMR_rank = pd.read_csv(
    mRMR_folder.joinpath(f"mRMR_ranks.csv"), header=None
)
mRMR_rank = mRMR_rank - 1

def get_rankings(_importances):
    n_features = len(_importances)

    # Sorted features from most important to least
    sortedfeatures = sorted(
        range(n_features), key=lambda i: _importances[i], reverse=True
    )

    # Rank of each feature
    feature_ranks = np.zeros(n_features)
    for r, f in enumerate(sortedfeatures):
        feature_ranks[f] = r # Index = Feature | Value = Rank

    return feature_ranks

# === === === ===
# Top features of IRelief
IRlf_ranksorted = IRlf_featurescores.apply(get_rankings)
IRlf_rank = pd.Series(data=np.zeros(ranks_df.shape[0]), name="IRlf")
for i, rank in enumerate(range(ranks_df.shape[0])):
    IRlf_rank[i] = IRlf_ranksorted[0][
        IRlf_ranksorted[0] == rank
    ].index[0]
ranks_df["IRlf"] = IRlf_rank

# === === === ===
# Top features of LHRelief
LHRlf_ranksorted = LHRlf_featurescores.apply(get_rankings)
LHRlf_rank = pd.Series(data=np.zeros(ranks_df.shape[0]), name="LHRlf")
for i, rank in enumerate(range(ranks_df.shape[0])):
    LHRlf_rank[i] = LHRlf_ranksorted[0][
        LHRlf_ranksorted[0] == rank
    ].index[0]
ranks_df["LHRlf"] = LHRlf_rank

# === === === ===
# Top features of mRMR

# Here as compared to the ranks df from IRlf and LHRlf,
#  - Index = Rank | Value = Feature
mRMR_rank = (mRMR_rank.T).loc[0:ranks_df.shape[0], 0]
ranks_df["mRMR"] = mRMR_rank

# === === === ===
# Reading scores from PDE-S
pdes_ranks = pd.read_csv(
    folder.joinpath(f"PDE-S/PDE-S_rank.csv"), index_col=0
)
ranks_df["PDE-S"] = pdes_ranks["PDE-S"]


# === === === ===
# Elapsed Times
py_times = pd.read_csv(
    folder.joinpath("OtherFS/elapsed_times.csv"), index_col=0
)['0']
IRlf_times = pd.read_csv(
    IRlf_folder.joinpath("tI.csv"), header=None
)
LHRlf_times = pd.read_csv(
    LHRlf_folder.joinpath("tLH.csv"), header=None
)
mRMR_times = pd.read_csv(
    mRMR_folder.joinpath("tmRMR.csv"), header=None
)
pdes_times = pd.read_csv(
    folder.joinpath("PDE-S/PDE-S_elapsed_times.csv"), index_col=0
)
py_times["IRlf"] = IRlf_times.values[0,0]
py_times["LHRlf"] = LHRlf_times.values[0,0]
py_times["mRMR"] = mRMR_times.values[0,0]
py_times["PDE-S"] = pdes_times.at["PDE-S",'0']

scores_df.to_csv(
    folder.joinpath("Combined/featurescores.csv")
)
ranks_df.to_csv(
    folder.joinpath("Combined/ranks.csv")
)
py_times.to_csv(
    folder.joinpath("Combined/elapsedtimes.csv")
)

sys.exit(0)
