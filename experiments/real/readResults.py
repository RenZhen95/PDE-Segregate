import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

if len(sys.argv) < 4:
    print(
        "Possible usage: python3 readResults.py <resultsFolder> " +
        "<datasetname> <xl_nslkdd>"
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])
    datasetname = sys.argv[2]
    xl_nslkdd = Path(sys.argv[3])

# Reading results for the following datasets
datasets = [
    "cns",
    "lung",
    "leuk",
    "colon",
    "pros3",
    "gcm",
    "geneExpressionCancerRNA",
    "PersonGaitDataSet"
]
# FSS name-mapper
FSS_dict = {
    "RlfF": "RlfF",
    "MSurf": "MSurf",
    "RlfI": "IRlf",
    "RlfLM": "LHRlf",
    "RFGini": "RFGini",
    "MI": "MI",
    "mRMR": "mRMR",
    "FT": "FT",
    "PDE-S": "PDE-S"
}

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S"
]

datasetResults_dict = defaultdict()

for f in os.scandir(resultsFolder):
    ds_name = (f.name).split('_')[0]
    datasetResults_dict[ds_name] = pd.read_csv(f, index_col=0)

    datasetResults_dict[ds_name]["FSS"] = [
        "" for i in range(datasetResults_dict[ds_name].shape[0])
    ]
    datasetResults_dict[ds_name]["nThreshold"] = [
        "" for i in range(datasetResults_dict[ds_name].shape[0])
    ]

    for i in datasetResults_dict[ds_name].index:
        fs = '-'.join(i.split('-')[:-1])
        th = i.split('-')[-1]
        if fs in list(FSS_dict.keys()):
            datasetResults_dict[ds_name].at[i, "FSS"] = FSS_dict[fs]
            datasetResults_dict[ds_name].at[i, "nThreshold"] = th
        else:
            datasetResults_dict[ds_name].drop([i], inplace=True)

    datasetResults_dict[ds_name].set_index("FSS", inplace=True)

overall_df = pd.DataFrame(
    data=np.zeros((len(fsorder), 5)), index=fsorder,
    columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
)

# Reading the individual dataset
df25 = datasetResults_dict[datasetname][
    datasetResults_dict[datasetname]["nThreshold"] == '25'
][["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]]
df25.loc["Average"] = df25.mean()

df50 = datasetResults_dict[datasetname][
    datasetResults_dict[datasetname]["nThreshold"] == '50'
][["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]]
df50.loc["Average"] = df50.mean()

df75 = datasetResults_dict[datasetname][
    datasetResults_dict[datasetname]["nThreshold"] == '75'
][["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]]
df75.loc["Average"] = df75.mean()

df100 = datasetResults_dict[datasetname][
    datasetResults_dict[datasetname]["nThreshold"] == '100'
][["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]]
df100.loc["Average"] = df100.mean()

print("Top 25"); print(df25.round(decimals=3).to_csv(sep=' '))
print("Top 50"); print(df50.round(decimals=3).to_csv(sep=' '))
print("Top 75"); print(df75.round(decimals=3).to_csv(sep=' '))
print("Top 100"); print(df100.round(decimals=3).to_csv(sep=' '))

# Read results from NSL-KDD dataset
nslkdd = pd.read_csv(xl_nslkdd, index_col=0)

# Computing the average of all datasets
print("Datasets included:")
for k in datasets:
    print(f" - {k}")
    df = datasetResults_dict[k].reset_index()
    df = df[["FSS", "kNN", "SVM", "Gaussian-NB", "LDA", "DT"]]

    dfmean = df.groupby("FSS").mean()
    dfmean = dfmean.loc[fsorder,:]

    overall_df += dfmean

print(" - NSL-KDD")

overall_df = overall_df + nslkdd.loc[fsorder]

nD = len(datasets) + 1
print(f"Total number of datasets: {nD}")
overall_mean_df = overall_df / nD

from ranktools import get_ranks_inplace
overall_mean_df_wRanks = overall_mean_df.apply(get_ranks_inplace)
overall_mean_df_wRanks = overall_mean_df_wRanks + 1
print(overall_mean_df_wRanks.to_csv(sep=' '))

overall_mean_df.loc["Average"] = overall_mean_df.mean()
print(overall_mean_df.round(decimals=3).to_csv(sep=' '))

sys.exit(0)
