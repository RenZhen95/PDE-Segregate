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
datasets = [
    ("cns", "CNS"),
    ("lung", "Lung"),
    ("leuk", "Leukemia"),
    ("colon", "Colon"),
    ("pros3", "Prostrate"),
    ("gcm", "GCM"),
    ("geneExpressionCancerRNA", "Cancer RNA-Gene"),
    ("PersonGaitDataSet", "Person Gait Classification")
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

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S"
]

overall_df = pd.DataFrame(
    data=np.zeros((len(fsorder), 5)), index=fsorder,
    columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
)

for k in datasets:
    print(k)
    df = datasetResults_dict[k[0]][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]
    print(df)
    dfmean = df.groupby("FSS").mean()
    dfmean = dfmean.loc[fsorder,:]
    print(dfmean)

    overall_df += dfmean

print(overall_df/len(datasets))

sys.exit(0)
