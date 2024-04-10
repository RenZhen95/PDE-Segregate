import pickle
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Reading the feature scores
with open("feature_scores.pkl", "rb" ) as handle:
    feature_scores = pickle.load(handle)

ANDOR_scoresdf = feature_scores["ANDOR"]
ADDER_scoresdf = feature_scores["ADDER"]

# Reading the top 10 features
with open("ranks.pkl", "rb") as handle:
    ranks = pickle.load(handle)

ANDOR_rankdf = ranks["ANDOR"]
ADDER_rankdf = ranks["ADDER"]

# Reading the elapsed times for each FSS method
elapsed_times_df = pd.read_csv("elapsed_times.csv", index_col=0)

# Features Summary
# ANDOR
# - Relevant   : 0, 1, 2, 3
# - Redundant  : 4, 5, 6, 7
# - Correlated : 8, 9
ANDOR_relevant   = [0, 1, 2, 3]
ANDOR_redundant  = [4, 5, 6, 7]
ANDOR_correlated = [8, 9]
# ADDER
# - Relevant   : 0, 1, 2
# - Redundant  : 3, 4, 5
# - Correlated : 6, 7

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

# groupeddf_count = ANDOR_rankdf.groupby(["iteration", "n_obs"]).apply(
#     count_instances_intop10, _list=ANDOR_relevant
# )
# relevant_top10count = groupeddf_count.reset_index("iteration")
# relevant_top10count = (relevant_top10count.groupby("n_obs")).mean()
# fss_list = list(relevant_top10count.columns)
# fss_list.remove("iteration")
# relevant_meancount_intop10 = pd.DataFrame(
#     data=np.zeros((len(fss_list)*3, 3)), columns=["Mean", "FSS", "n_obs"]
# )
# i = 0
# for fss in fss_list:
#     for n in relevant_top10count.index:
#         relevant_meancount_intop10.at[i, "Mean"] = relevant_top10count.at[n, fss]
#         relevant_meancount_intop10.at[i, "FSS"] = fss
#         relevant_meancount_intop10.at[i, "n_obs"] = n
#         i += 1

# print(relevant_meancount_intop10)
# sns.barplot(data=relevant_meancount_intop10, x="FSS", y="Mean", hue="n_obs")
# plt.tight_layout()

# Ranking features
def get_rankings(_importances):
    list_FSS = list(_importances.columns)
    list_FSS.remove("feature")
    list_FSS.remove("iteration")
    list_FSS.remove("n_obs")

    n_features = _importances.shape[0]
    rank = pd.DataFrame(
        data=np.zeros((n_features, len(list_FSS))), columns=list_FSS
    )
    for fss in list_FSS:
        feature_importance = _importances[fss].reset_index(drop=True)

        # Sorted features from most important to least
        sortedfeatures = sorted(
            range(n_features), key=lambda i: feature_importance[i], reverse=True
        )

        # Rank of each feature
        feature_ranks = np.zeros(n_features)
        for r, f in enumerate(sortedfeatures):
            feature_ranks[f] = r

        rank[fss] = feature_ranks

    return rank

groupeddf_ranks = ANDOR_scoresdf.groupby(["iteration", "n_obs"]).apply(
    get_rankings
)
print(groupeddf_ranks)
groupeddf_ranks.to_csv("rmlater.csv")

groupeddf_ranks_n30 = groupeddf_ranks.xs(30, level=1)
groupeddf_ranks_n50 = groupeddf_ranks.xs(50, level=1)
groupeddf_ranks_n70 = groupeddf_ranks.xs(70, level=1)

ranks_n30_f0 = groupeddf_ranks_n30.xs(0, level=1)
ranks_n50_f0 = groupeddf_ranks_n50.xs(0, level=1)
ranks_n70_f0 = groupeddf_ranks_n70.xs(0, level=1)

# print(ranks_n30_f0.median())
# print(ranks_n50_f0.median())
# print(ranks_n70_f0.median())

#plt.show()
sys.exit()
