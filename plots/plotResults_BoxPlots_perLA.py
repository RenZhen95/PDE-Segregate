import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

def plot_boxPlot(df, datasetName, algo, ax, marker="*", s=15):
    # 4 thresholds x nFSS
    df_boxPlot = pd.DataFrame(
        data=np.zeros((4*len(FSS_dict), 3)),
        columns=["Bal. Acc", "FSS", "# Top Features Retained"]
    )
    df_boxPlot = df_boxPlot.astype({"FSS": "object"})

    count = 0
    for i in df.index:
        nThreshold = int(i.split('-')[1])
        df_boxPlot.at[count, "Bal. Acc"] = df.at[i, algo]
        df_boxPlot.at[count, "FSS"] = df.at[i, "FSS"]
        df_boxPlot.at[count, "# Top Features Retained"] = nThreshold
        count += 1

    axStripPlot = sns.stripplot(
        data=df_boxPlot, x="Bal. Acc", y="FSS", legend=False,
        hue="# Top Features Retained", s=s, marker=marker,
        palette="coolwarm", alpha=0.55, ax=ax
    )

    axStripPlot.set_xlabel("Balanced Accuracy", fontsize="large")
    axStripPlot.set_xlim(0.45, 1.02)
    axStripPlot.set_xticks(np.arange(0.5, 1.05, 0.1))

    axStripPlot.set_ylabel("")
    
    axStripPlot.tick_params(axis="both", labelsize="large")
    axStripPlot.set_title(datasetName, fontsize="x-large")

    axStripPlot.grid(visible=True)

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
    "geneExpressionCancerRNA": "Cancer RNA-Gene Expression",
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
figkNN, axskNN = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figSVM, axsSVM = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figGNB, axsGNB = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figLDA, axsLDA = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figDT, axsDT = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
axisCounter = 0
for d, df in datasetResults_dict.items():
    plot_boxPlot(
        df, d, "kNN", axskNN[int(axisCounter/3), axisCounter%3]
    )
    plot_boxPlot(
        df, d, "SVM", axsSVM[int(axisCounter/3), axisCounter%3], s=12, marker="X"
    )
    plot_boxPlot(
        df, d, "Gaussian-NB", axsGNB[int(axisCounter/3), axisCounter%3], s=12, marker="^"
    )
    plot_boxPlot(
        df, d, "LDA", axsLDA[int(axisCounter/3), axisCounter%3], s=12, marker=9
    )
    plot_boxPlot(
        df, d, "DT", axsDT[int(axisCounter/3), axisCounter%3], s=12, marker="P"
    )
    axisCounter += 1

figkNN.suptitle(
    "k-Nearest-Neighbor Classifier", fontsize="xx-large",
    x=0.022, y=0.97, horizontalalignment="left"
)
figkNN.tight_layout()

figSVM.suptitle(
    r"Support Vector Classifier", fontsize="xx-large",
    x=0.022, y=0.97, horizontalalignment="left"
)
figSVM.tight_layout()

figGNB.suptitle(
    r"Gaussian Naive Bayes Classifier", fontsize="xx-large",
    x=0.022, y=0.97, horizontalalignment="left"
)
figGNB.tight_layout()

figLDA.suptitle(
    r"Linear Discriminant Classifier", fontsize="xx-large",
    x=0.022, y=0.97, horizontalalignment="left"
)
figLDA.tight_layout()

figDT.suptitle(
    r"Decision Tree Classifier", fontsize="xx-large",
    x=0.022, y=0.97, horizontalalignment="left"
)
figDT.tight_layout()

plt.show()
sys.exit(0)
