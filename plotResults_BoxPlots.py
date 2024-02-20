import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

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

    df_boxPlot25 = df_boxPlot[df_boxPlot["# Top Features Retained"] == 25]
    df_boxPlot25.reset_index(inplace=True)
    df_boxPlot50 = df_boxPlot[df_boxPlot["# Top Features Retained"] == 50]
    df_boxPlot50.reset_index(inplace=True)
    df_boxPlot75 = df_boxPlot[df_boxPlot["# Top Features Retained"] == 75]
    df_boxPlot75.reset_index(inplace=True)
    df_boxPlot100 = df_boxPlot[df_boxPlot["# Top Features Retained"] == 100]
    df_boxPlot100.reset_index(inplace=True)

    axBoxPlot = sns.boxplot(
        data=df_boxPlot, x="Bal. Acc", y="FSS",
        ax=ax, width=0.7, fliersize=0., fill=False
    )

    _alpha = 0.75
    sns.stripplot(
        data=df_boxPlot25, x="Bal. Acc", y="FSS",
        hue="LAlgo", ax=ax, marker="o", linewidth=1, alpha=_alpha,
        legend=False
    )
    sns.stripplot(
        data=df_boxPlot50, x="Bal. Acc", y="FSS",
        hue="LAlgo", ax=ax, marker="*", s=10, linewidth=1, alpha=_alpha,
        legend=False
    )
    sns.stripplot(
        data=df_boxPlot75, x="Bal. Acc", y="FSS",
        hue="LAlgo", ax=ax, marker="X", s=6.5, linewidth=1, alpha=_alpha,
        legend=False
    )
    sns.stripplot(
        data=df_boxPlot100, x="Bal. Acc", y="FSS",
        hue="LAlgo", ax=ax, marker="P", s=7.5, linewidth=1, alpha=_alpha,
        legend=False
    )

    axBoxPlot.set_xlabel("Balanced Accuracy", fontsize="large")
    axBoxPlot.set_xlim(0.48, 1.02)
    axBoxPlot.set_xticks(np.arange(0.5, 1.05, 0.1))

    axBoxPlot.set_ylabel("")
    
    axBoxPlot.tick_params(axis="both", labelsize="large")
    axBoxPlot.set_title(datasetName, fontsize="x-large")

    return axBoxPlot, df_boxPlot

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
    "dlbcl": "DLBCL",
    "pros3": "Prostrate",
    "geneExpressionCancerRNA": "Cancer RNA-Gene Expression",
    "colon": "Colon",
    "gcm": "GCM"
    #"PersonGaitDataSet": "Person Gait"
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
    "EtaSq": r"ANOVA $\eta^2$",
    "OAscott": "PDE-Segregate (Scott)",
    "OAsilverman": "PDE-Segregate (Silverman)",
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
# fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharey=True, sharex=True)
axisCounter = 0
for d, df in datasetResults_dict.items():
    axBoxPlot, df_forBoxPlots = plot_boxPlot(
        df, d, axs[int(axisCounter/3), axisCounter%3]
    )
    axisCounter += 1        

fig.tight_layout()
plt.show()
sys.exit(0)
