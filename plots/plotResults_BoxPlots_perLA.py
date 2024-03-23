import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

def plot_boxPlot(df, datasetName, algo, ax, marker="*", s=15, xlim="default"):
    # 4 thresholds x nFSS
    nFSS = int(df.shape[0]/4)
    df_boxPlot = pd.DataFrame(
        data=np.zeros((4*nFSS, 3)),
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
    if xlim == "default":
        axStripPlot.set_xlim(0.45, 1.02)
        axStripPlot.set_xticks(np.arange(0.5, 1.05, 0.1))
    else:
        axStripPlot.set_xlim(xlim)

    axStripPlot.set_ylabel("")
    
    axStripPlot.tick_params(axis="both", labelsize="large")
    axStripPlot.set_title(datasetName, fontsize="x-large")

    axStripPlot.grid(visible=True)

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
    ("gcm", "GCM")
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

datasetResults_binary = defaultdict()
for k in datasetResults_dict.keys():
    df = datasetResults_dict[k]
    df = df.drop([i for i in list(df.index) if "OApw" in i])
    datasetResults_binary[k] = df

# Plot
figkNN, axskNN = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figSVM, axsSVM = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figGNB, axsGNB = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figLDA, axsLDA = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
figDT, axsDT = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
axisCounter = 0
for i in range(len(datasets)):
    df = datasetResults_binary[datasets[i][0]]
    d  = datasets[i][1]
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

# figkNN.suptitle(
#     "k-Nearest-Neighbor Classifier", fontsize="xx-large",
#     x=0.022, y=0.97, horizontalalignment="left"
# )

# figSVM.suptitle(
#     r"Support Vector Classifier", fontsize="xx-large",
#     x=0.022, y=0.97, horizontalalignment="left"
# )

# figGNB.suptitle(
#     r"Gaussian Naive Bayes Classifier", fontsize="xx-large",
#     x=0.022, y=0.97, horizontalalignment="left"
# )

# figLDA.suptitle(
#     r"Linear Discriminant Classifier", fontsize="xx-large",
#     x=0.022, y=0.97, horizontalalignment="left"
# )

# figDT.suptitle(
#     r"Decision Tree Classifier", fontsize="xx-large",
#     x=0.022, y=0.97, horizontalalignment="left"
# )
figkNN.tight_layout()
figSVM.tight_layout()
figGNB.tight_layout()
figLDA.tight_layout()
figDT.tight_layout()

figkNN.savefig("kNN.png", format="png")
figSVM.savefig("SVM.png", format="png")
figGNB.savefig("GNB.png", format="png")
figLDA.savefig("LDA.png", format="png")
figDT.savefig("DT.png", format="png")

# # Multiclass Datasets
# figkNN_mc, axkNN_mc = plt.subplots(1, 2, figsize=(11.5, 4.75), sharey=True)
# figSVM_mc, axSVM_mc = plt.subplots(1, 2, figsize=(11.5, 4.75), sharey=True)
# figGNB_mc, axGNB_mc = plt.subplots(1, 2, figsize=(11.5, 4.75), sharey=True)
# figLDA_mc, axLDA_mc = plt.subplots(1, 2, figsize=(11.5, 4.75), sharey=True)
# figDT_mc, axDT_mc = plt.subplots(1, 2, figsize=(11.5, 4.75), sharey=True)

# datasetResults_multi = {}
# datasetResults_multi["geneExpressionCancerRNA"] = datasetResults_dict[
#     "geneExpressionCancerRNA"
# ].map(
#     lambda x: "PDE-Segregate (A)" if x=="PDE-Segregate" else x
# )
# datasetResults_multi["PersonGaitDataSet"] = datasetResults_dict[
#     "PersonGaitDataSet"
# ].map(
#     lambda x: "PDE-Segregate (A)" if x=="PDE-Segregate" else x
# )
# dataset_multi_name = {
#     "geneExpressionCancerRNA": "Cancer RNA-Gene Expression",
#     "PersonGaitDataSet": "Person Gait"
# }
# for i, k in enumerate(datasetResults_multi.keys()):
#     plot_boxPlot(
#         datasetResults_multi[k], dataset_multi_name[k],
#         "kNN", axkNN_mc[i], xlim=(-0.02, 1.02)
#     )
#     plot_boxPlot(
#         datasetResults_multi[k], dataset_multi_name[k],
#         "SVM", axSVM_mc[i], xlim=(-0.02, 1.02), s=12, marker="X"
#     )
#     plot_boxPlot(
#         datasetResults_multi[k], dataset_multi_name[k],
#         "Gaussian-NB", axGNB_mc[i], xlim=(-0.02, 1.02), s=12, marker="^"
#     )
#     plot_boxPlot(
#         datasetResults_multi[k], dataset_multi_name[k],
#         "LDA", axLDA_mc[i], xlim=(-0.02, 1.02), s=12, marker=9
#     )
#     plot_boxPlot(
#         datasetResults_multi[k], dataset_multi_name[k],
#         "DT", axDT_mc[i], xlim=(-0.02, 1.02), s=12, marker="P"
#     )

# figkNN_mc.tight_layout()
# figSVM_mc.tight_layout()
# figGNB_mc.tight_layout()
# figLDA_mc.tight_layout()
# figDT_mc.tight_layout()

# figkNN_mc.savefig("mc_kNN.png", format="png")
# figSVM_mc.savefig("mc_SVM.png", format="png")
# figGNB_mc.savefig("mc_GNB.png", format="png")
# figLDA_mc.savefig("mc_LDA.png", format="png")
# figDT_mc.savefig("mc_DT.png", format="png")

# plt.show()

sys.exit(0)
