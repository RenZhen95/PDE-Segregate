import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

def plot_boxPlot(df, datasetName, ax):
    df_boxPlot = pd.DataFrame(
        data=np.zeros((6*4*len(FSS_dict),3)),
        columns=["Bal. Acc", "FSS", "LAlgo"]
    )
    df_boxPlot = df_boxPlot.astype({"FSS": "object", "LAlgo": "object"})

    LAlgos = ["kNN", "SVM", "Gaussian-NB", "LDA"]
    count = 0
    for i in df.index:
        for algo in LAlgos:
            df_boxPlot.at[count, "Bal. Acc"] = df.at[i, algo]
            df_boxPlot.at[count, "FSS"] = df.at[i, "FSS"]
            df_boxPlot.at[count, "LAlgo"] = algo
            count += 1
  
    axBoxPlot = sns.boxplot(
        data=df_boxPlot, x="Bal. Acc", y="FSS",
        ax=ax, width=0.7, fliersize=0., fill=False, legend=False
    )
    sns.stripplot(
        data=df_boxPlot, x="Bal. Acc", y="FSS",
        hue="LAlgo", ax=ax, legend=False
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
    "dlbcl": "DLBCL",
    "lung": "Lung",
    "leuk": "Leukemia",
    "pros3": "Prostrate",
    "geneExpressionCancerRNA": "Cancer RNA-Gene Expression"
}
# FSS name-mapper
FSS_dict = {
    "RlfF": "RELIEF-F",
    "MSurf": "MultiSURF",
    "RlfI": "I-RELIEF",
    "RlfLM": "LH-RELIEF",
    "MI": "Mutual Information",
    "FT": "ANOVA (F-Statistic)",
    "RFGini": "Random Forest (Gini)",
#   "RFEtry": "Entropy (RF)",
    "OA": "PDE-Segregate"
}

# 6 benchmark datasets
datasetResults_dict = defaultdict()
for d, data_name in datasets.items():
    datasetResults_dict[data_name] = pd.read_excel(
        resultsXlFile, sheet_name=d, index_col=0, usecols="A:E"
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
    axBoxPlot, df_forBoxPlots = plot_boxPlot(df, d, axs[int(axisCounter/3), axisCounter%3])
    axisCounter += 1        

fig.tight_layout()
plt.show()
sys.exit(0)
