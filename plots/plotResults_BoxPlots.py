import os, sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

def get_best_perLA(df, learningAlgo):
    learningAlgo_df = df[df["LAlgo"]==learningAlgo]
    top_df = learningAlgo_df[
        learningAlgo_df["Bal. Acc"]==learningAlgo_df["Bal. Acc"].max()
    ]
    top_df.reset_index(inplace=True, drop=True)

    topFSS = top_df["FSS"]
    topFSS_set = list(set(list(topFSS)))

    # To deal with the case, where multiple FSS produce top performance
    if len(topFSS) > 1:
        print(f"{learningAlgo} shows best performance with multiple FSS")
        prepped_top_df = pd.DataFrame(
            columns=["Bal. Acc", "FSS", "LAlgo", "# Top Features Retained"]
        )
        for fss in topFSS_set:
            prepped_top_df = pd.concat(
                [prepped_top_df, top_df[top_df["FSS"]==fss].iloc[0:1]]
            )
    else:
        prepped_top_df = top_df

    return top_df

def plot_boxPlot(df, datasetName, ax, nLA=5, set_xlim=None):
    # 4 thresholds x 5 learning algo x nFSS
    nFSS = int(df.shape[0]/4)
    df_boxPlot = pd.DataFrame(
        data=np.zeros((4*nLA*nFSS, 4)),
        columns=["Bal. Acc", "FSS", "LAlgo", "# Top Features Retained"]
    )
    df_boxPlot = df_boxPlot.astype({"FSS": "object", "LAlgo": "object"})
    LAlgos = [i for i in df.columns if i != "FSS"]
    count = 0
    for i in df.index:
        nThreshold = int(i.split('-')[1])
        for algo in LAlgos:
            df_boxPlot.at[count, "Bal. Acc"] = df.at[i, algo]
            df_boxPlot.at[count, "FSS"] = df.at[i, "FSS"]
            df_boxPlot.at[count, "LAlgo"] = algo
            df_boxPlot.at[count, "# Top Features Retained"] = nThreshold
            count += 1

    axBoxPlot = sns.boxplot(
        data=df_boxPlot, x="Bal. Acc", y="FSS",
        ax=ax, width=0.7, fliersize=0., fill=False
    )

    # Plot points of the best performance per LA
    topkNN = get_best_perLA(df_boxPlot, "kNN")
    topSVM = get_best_perLA(df_boxPlot, "SVM")
    topGNB = get_best_perLA(df_boxPlot, "Gaussian-NB")
    topLDA = get_best_perLA(df_boxPlot, "LDA")
    topDT = get_best_perLA(df_boxPlot, "DT")

    _alpha = 0.55
    _color = "gray"
    sns.stripplot(
        data=topkNN, x="Bal. Acc", y="FSS",
        ax=ax, marker="*", s=15, linewidth=1, alpha=_alpha,
        color=_color, legend=False
    )
    sns.stripplot(
        data=topSVM, x="Bal. Acc", y="FSS",
        ax=ax, marker="X", s=12, linewidth=1, alpha=_alpha,
        color=_color, legend=False
    )
    sns.stripplot(
        data=topGNB, x="Bal. Acc", y="FSS",
        ax=ax, marker="^", s=12, linewidth=1, alpha=_alpha,
        color=_color, legend=False
    )
    sns.stripplot(
        data=topLDA, x="Bal. Acc", y="FSS",
        ax=ax, marker=9, s=12, linewidth=1, alpha=_alpha,
        color=_color, legend=False
    )
    sns.stripplot(
        data=topDT, x="Bal. Acc", y="FSS",
        ax=ax, marker="P", s=12, linewidth=1, alpha=_alpha,
        color=_color, legend=False
    )

    axBoxPlot.set_xlabel("Balanced Accuracy", fontsize="large")
    if set_xlim is None:
        axBoxPlot.set_xlim(0.45, 1.02)
        axBoxPlot.set_xticks(np.arange(0.5, 1.05, 0.1))

    axBoxPlot.set_ylabel("")

    axBoxPlot.tick_params(axis="both", labelsize="large")
    axBoxPlot.set_title(datasetName, fontsize="x-large")

    axBoxPlot.grid(visible=True)

if len(sys.argv) < 2:
    print("Possible usage: python3 plotResults_BoxPlots.py <resultsFolder>")
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])

# Reading results for the following datasets
datasets = [
    ("cns", "CNS"),
    ("lung", "Lung"),
    ("leuk", "Leukemia"),
    ("colon", "Colon"),
    ("pros3", "Prostrate"),
    ("gcm", "GCM"),
    ("dlbcl", "DLBCL"),
    ("PersonGaitDataSet", "Person Gait"),
    ("geneExpressionCancerRNA", "Cancer RNA-Gene Expression"),
    ("pros1", "Prostrate 1"),
    ("pros2", "Prostrate 2")
]
# FSS name-mapper
FSS_dict = {
    "RlfF": "RELIEF-F",
    "MSurf": "MultiSURF",
    "RlfI": "I-RELIEF",
    "RlfLM": "LH-RELIEF",
    "RFGini": "Random Forest (Gini)",
    "MI": "Mutual Information",
    "FT": "ANOVA (F-Statistic)",
    "OAtotal": "PDE-Segregate",
    "OApw": "PDE-Segregate (B)"
}

# 6 benchmark datasets
datasetResults_dict = defaultdict()

for f in os.scandir(resultsFolder):
    ds_name = (f.name).split('_')[0]
    datasetResults_dict[ds_name] = pd.read_csv(f, index_col=0)

    datasetResults_dict[ds_name]["FSS"] = [
        "" for i in range(datasetResults_dict[ds_name].shape[0])
    ]
    for i in datasetResults_dict[ds_name].index:
        if i.split('-')[0] in list(FSS_dict.keys()):
            datasetResults_dict[ds_name].at[i, "FSS"] = FSS_dict[i.split('-')[0]]
        else:
            datasetResults_dict[ds_name].drop([i], inplace=True)

datasetResultsMulti_dict = datasetResults_dict.copy()
for k in datasetResults_dict.keys():
    df = datasetResults_dict[k]
    df = df.drop([i for i in list(df.index) if "OApw" in i])
    datasetResults_dict[k] = df

# Plot
fig, axs   = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
for i in range(len(datasets)):
    if i < 6:
        plot_boxPlot(
            datasetResults_dict[datasets[i][0]],
            datasets[i][1], axs[int(i/3), i%3]
        )
    else:
        plot_boxPlot(
            datasetResults_dict[datasets[i][0]],
            datasets[i][1], axs2[int((i-6)/3), (i-6)%3]
        )

fig.tight_layout()
fig2.tight_layout()

# Multiclass Datasets
fig3, axs3 = plt.subplots(1, 2, figsize=(11.5, 4.75), sharey=True)
multiclass_datasets = [
    ("geneExpressionCancerRNA", "Cancer RNA-Gene Expression"),
    ("PersonGaitDataSet", "Person Gait")
]
datasetResultsMulti_dict["geneExpressionCancerRNA"] = datasetResultsMulti_dict[
    "geneExpressionCancerRNA"
].map(
    lambda x: "PDE-Segregate (A)" if x=="PDE-Segregate" else x
)
datasetResultsMulti_dict["PersonGaitDataSet"] = datasetResultsMulti_dict[
    "PersonGaitDataSet"
].map(
    lambda x: "PDE-Segregate (A)" if x=="PDE-Segregate" else x
)
# For the multiclass datasets, only take SVM and LDA
# Cancer Gene
reduced_cancer = datasetResultsMulti_dict["geneExpressionCancerRNA"]
reduced_cancer.drop(["kNN", "Gaussian-NB", "DT"], axis=1, inplace=True)
# Person Gait
reduced_persongait = datasetResultsMulti_dict["PersonGaitDataSet"]
reduced_persongait.drop(["kNN", "Gaussian-NB", "DT"], axis=1, inplace=True)

plot_boxPlot(
    reduced_cancer, multiclass_datasets[0][1], axs3[0], nLA=2
)

plot_boxPlot(
    reduced_persongait, multiclass_datasets[1][1], axs3[1], nLA=2
)
fig3.tight_layout()

plt.show()

sys.exit(0)
