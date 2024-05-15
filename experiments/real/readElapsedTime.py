import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

if len(sys.argv) < 4:
    print(
        "Possible usage: python3 readElapsedTime.py <pyTime> " +
        "<matlabfs_folder> <nslkdd_times>"
    )
    sys.exit()
else:
    pytime = Path(sys.argv[1])
    matlabfs_folder = Path(sys.argv[2])
    nslkddtimes = Path(sys.argv[3])

with open(pytime, "rb") as handle:
    pytimes = pickle.load(handle)

IRlf = pd.read_csv(
    matlabfs_folder.joinpath("IRelief_timenoANOVAcutclean.csv"),
    header=None, index_col=0
)
LHRlf = pd.read_csv(
    matlabfs_folder.joinpath("LHRelief_timenoANOVAcutclean.csv"),
    header=None, index_col=0
)
mRMR = pd.read_csv(
    matlabfs_folder.joinpath("mRMR_timenoANOVAcutclean.csv"),
    header=None, index_col=0
)

# Combining the computational times across real datasets
comptimes = pytimes.copy()
for ds in comptimes.keys():
    comptimes[ds].update({"IRlf": IRlf.at[ds, 1]})
    comptimes[ds].update({"LHRlf": LHRlf.at[ds, 1]})
    comptimes[ds].update({"mRMR": mRMR.at[ds, 1]})

# Combining the computational times for the NSL-KDD dataset
nslkdd_times = pd.read_csv(nslkddtimes, index_col=0)
nslkdd_times.columns = ["NSL-KDD"]
comptimes.update({"NSL-KDD": nslkdd_times.to_dict()["NSL-KDD"]})

# Reading the elapsed times
FSS = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini",  "MI",
    "mRMR", "FT",
    "PDE-S"
]

datasets = [
    "cns", "lung", "leuk",
    "colon", "pros3", "gcm",
    "geneExpressionCancerRNA", "PersonGaitDataSet",
    "NSL-KDD"
]

elapsedTimes_df = pd.DataFrame(
    data=np.zeros((len(datasets), len(FSS))),
    index=datasets, columns=FSS
)

for fss in FSS:
    for ds in datasets:
        elapsedTimes_df.at[ds, fss] = comptimes[ds][fss]

print(elapsedTimes_df.round(decimals=2))
print(elapsedTimes_df.mean())
print("\nAverage of all average computational times:")
print((elapsedTimes_df.mean()).mean())
print(elapsedTimes_df.to_latex(index=False, float_format="%.3f"))

sys.exit(0)
