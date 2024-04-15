import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(
        "Possible usage: python3.11 evaluate_fss.py <resultsFolder> <datasetName>"
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])
    datasetName = sys.argv[2]
    
# Reading the feature scores
feature_scores_df = pd.read_csv(
    resultsFolder.joinpath(f"{datasetName}_featurescores.csv"), index_col=0
)

# Reading the top 10 features
ranks_df = pd.read_csv(
    resultsFolder.joinpath(f"{datasetName}_ranks.csv"), index_col=0
)

# Reading the elapsed times for each FSS method
elapsed_times_df = pd.read_csv(
    resultsFolder.joinpath(f"{datasetName}_elapsedtimes.csv"), index_col=0
)

# Features Summary
# ANDOR
# - Relevant   : 0, 1, 2, 3
# - Redundant  : 4, 5, 6, 7
# - Correlated : 8, 9
if datasetName[0:5] == "ANDOR":
    relevant_list   = [0, 1, 2, 3]
    redundant_list  = [4, 5, 6, 7]
    correlated_list = [8, 9]
# ADDER
# - Relevant   : 0, 1, 2
# - Redundant  : 3, 4, 5
# - Correlated : 6, 7
elif datasetName[0:5] == "ADDER":
    relevant_list   = [0, 1, 2]
    redundant_list  = [3, 4, 5]
    correlated_list = [6, 7]

def count_instances_intop10(_top10, _list=[]):
    list_FSS = list(_top10.columns)
    list_FSS.remove("rank")
    list_FSS.remove("iteration")
    list_FSS.remove("n_obs")

    count = pd.Series(data=np.zeros(len(list_FSS)), index=list_FSS)

    for i in _top10.index:
        for fss in count.index:
            if _top10.at[i, fss] in _list:
                count.at[fss] += 1

    return count

groupeddf_relvcount = ranks_df.groupby(["iteration", "n_obs"]).apply(
    count_instances_intop10, _list=relevant_list
)

def check_ifrelevantistop(_top10, _list=[]):
    list_FSS = list(_top10.columns)
    list_FSS.remove("rank")
    list_FSS.remove("iteration")
    list_FSS.remove("n_obs")

    _top10_nfirst = _top10.iloc[:len(_list),:]
    relv_atthetop = pd.Series(
        data=[False for i in range(len(list_FSS))], index=list_FSS
    )

    for fss in list_FSS:
        if set(_top10_nfirst[fss].values) == set(_list):
            relv_atthetop[fss] = True
            
    return relv_atthetop

groupeddf_relvattop = ranks_df.groupby(["iteration", "n_obs"]).apply(
    check_ifrelevantistop, _list=relevant_list
)

# Computing the success rate as defined by Canedo, 2012
print(groupeddf_relvcount)
print(groupeddf_relvattop)

fss_list = list(groupeddf_relvattop.columns)
relvcount = groupeddf_relvcount.xs(30, level=1)
relvattop = groupeddf_relvattop.xs(30, level=1)

average_successrate = pd.DataFrame(
    data=np.zeros((3, len(fss_list))), index=[30, 50, 70], columns=fss_list
)
print(average_successrate)
# Metrics according to Kamalov, 2022
# 1. Simply compare feature significance metric of relevant vs non-relevant features
# 2. Median rankings across 10 iterations of relevant features

# # Ranking features (Implementation of 2.)
# def get_rankings(_importances):
#     list_FSS = list(_importances.columns)
#     list_FSS.remove("feature")
#     list_FSS.remove("iteration")
#     list_FSS.remove("n_obs")

#     n_features = _importances.shape[0]
#     rank = pd.DataFrame(
#         data=np.zeros((n_features, len(list_FSS))), columns=list_FSS
#     )
#     for fss in list_FSS:
#         feature_importance = _importances[fss].reset_index(drop=True)

#         # Sorted features from most important to least
#         sortedfeatures = sorted(
#             range(n_features), key=lambda i: feature_importance[i], reverse=True
#         )

#         # Rank of each feature
#         feature_ranks = np.zeros(n_features)
#         for r, f in enumerate(sortedfeatures):
#             feature_ranks[f] = r

#         rank[fss] = feature_ranks

#     return rank

# groupeddf_ranks = ANDOR_scoresdf.groupby(["iteration", "n_obs"]).apply(
#     get_rankings
# )
# print(groupeddf_ranks)

# groupeddf_ranks_n30 = groupeddf_ranks.xs(30, level=1)
# groupeddf_ranks_n50 = groupeddf_ranks.xs(50, level=1)
# groupeddf_ranks_n70 = groupeddf_ranks.xs(70, level=1)

# ranks_n30_f0 = groupeddf_ranks_n30.xs(0, level=1)
# ranks_n50_f0 = groupeddf_ranks_n50.xs(0, level=1)
# ranks_n70_f0 = groupeddf_ranks_n70.xs(0, level=1)

# print(ranks_n30_f0.median())
# print(ranks_n50_f0.median())
# print(ranks_n70_f0.median())

# plt.show()

sys.exit()
