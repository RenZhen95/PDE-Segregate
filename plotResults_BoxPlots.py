import os, sys
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

def plot_boxPlot(df, datasetName, ax):
    # 4 thresholds x 5 learning algo x nFSS
    df_boxPlot = pd.DataFrame(
        data=np.zeros((4*5*len(FSS_dict), 4)),
        columns=["Bal. Acc", "FSS", "LAlgo", "# Top Features Retained"]
    )
    df_boxPlot = df_boxPlot.astype({"FSS": "object", "LAlgo": "object"})

    LAlgos = ["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
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
    axBoxPlot.set_xlim(0.45, 1.02)
    axBoxPlot.set_xticks(np.arange(0.5, 1.05, 0.1))

    axBoxPlot.set_ylabel("")
    
    axBoxPlot.tick_params(axis="both", labelsize="large")
    axBoxPlot.set_title(datasetName, fontsize="x-large")

    axBoxPlot.grid(visible=True)

if len(sys.argv) < 2:
    print("Possible usage: python3 plotResults_BoxPlots.py <resultsXlFile>")
    sys.exit(1)
else:
    resultsXlFile = Path(sys.argv[1])

# Reading results for the following datasets
datasets = {
    "cns": "CNS",
    "lung": "Lung",
    "leuk": "Leukemia",
    "colon": "Colon",
    "pros3": "Prostrate",
    "geneExpressionCancerRNA": "Cancer RNA-Gene Expression"
    # "gcm": "GCM"
    # "dlbcl": "DLBCL",
}
# FSS name-mapper
FSS_dict = {
    "RlfF": "RELIEF-F",
    "MSurf": "MultiSURF",
    "RlfI": "I-RELIEF",
    "RlfLM": "LH-RELIEF",
    "RFGini": "Random Forest (Gini)",
    "MI": "Mutual Information",
    "FT": "ANOVA (F-Statistic)",
    "OAscott": "PDE-Segregate"
}

# 6 benchmark datasets
datasetResults_dict = defaultdict()
for d, data_name in datasets.items():
    datasetResults_dict[data_name] = pd.read_excel(
        resultsXlFile, sheet_name=d, index_col=0, usecols="A:F"
    )
    datasetResults_dict[data_name]["FSS"] = [
        "" for i in range(datasetResults_dict[data_name].shape[0])
    ]
    for i in datasetResults_dict[data_name].index:
        if i.split('-')[0] in list(FSS_dict.keys()):
            datasetResults_dict[data_name].at[i, "FSS"] = FSS_dict[i.split('-')[0]]
        else:
            datasetResults_dict[data_name].drop([i], inplace=True)

# Plot
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
axisCounter = 0
for d, df in datasetResults_dict.items():
    print(f"=== === ===\n{d}\n=== === ===")
    plot_boxPlot(
        df, d, axs[int(axisCounter/3), axisCounter%3]
    )
    axisCounter += 1

fig.suptitle(
    "Best Performance per Classifier", fontsize="xx-large",
    x=0.022, y=0.97, horizontalalignment="left"
)
fig.tight_layout()

plt.show()

sys.exit(0)
