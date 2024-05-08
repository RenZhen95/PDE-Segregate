import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

if len(sys.argv) < 2:
    print("Possible usage: python3 readResults.py <resultsFolder>")
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])

# Reading results for the following datasets
# datasets_binary = [
#     ("cns", "CNS"),
#     ("lung", "Lung"),
#     ("leuk", "Leukemia"),
#     ("colon", "Colon"),
#     ("pros3", "Prostrate"),
#     ("gcm", "GCM")
# ]
datasets_binary = [
    ("dlbcl", "DLBCL"),
    ("pros1", "Prostrate1"),
    ("pros2", "Prostrate2")
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
    "OAtotal": "PDE-S",
    "OApw": "PDE-S*"
}

# 6 benchmark datasets
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
        fs = i.split('-')[0]
        th = i.split('-')[1]
        if fs in list(FSS_dict.keys()):
            datasetResults_dict[ds_name].at[i, "FSS"] = FSS_dict[fs]
            datasetResults_dict[ds_name].at[i, "nThreshold"] = th
        else:
            datasetResults_dict[ds_name].drop([i], inplace=True)

datasetResults_binary = defaultdict()

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S"
]

for k in datasets_binary:
    print(k)
    df = datasetResults_dict[k[0]]
    df = df.drop([i for i in list(df.index) if "OApw" in i])

    df25  = df[df["nThreshold"]=="25"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df25 = df25.set_index("FSS")
    df25 = df25.loc[fsorder,:]
    df25.loc["Average", :] = df25.mean()

    df50  = df[df["nThreshold"]=="50"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df50 = df50.set_index("FSS")
    df50 = df50.loc[fsorder,:]
    df50.loc["Average", :] = df50.mean()

    df75  = df[df["nThreshold"]=="75"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df75 = df75.set_index("FSS")
    df75 = df75.loc[fsorder,:]
    df75.loc["Average", :] = df75.mean()

    df100 = df[df["nThreshold"]=="100"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df100 = df100.set_index("FSS")
    df100 = df100.loc[fsorder,:]
    df100.loc["Average", :] = df100.mean()

    print(df25)
    print(df25.to_latex(float_format="%.3f", index=False))
    print(df50)
    print(df50.to_latex(float_format="%.3f", index=False))
    print(df75)
    print(df75.to_latex(float_format="%.3f", index=False))
    print(df100)
    print(df100.to_latex(float_format="%.3f", index=False))
    
    input("")

sys.exit(0)
