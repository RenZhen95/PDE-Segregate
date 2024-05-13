import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

if len(sys.argv) < 6:
    print(
        "Possible usage: python3 readElapsedTime.py <pyTime> " +
        "<IRlf> <LHRlf> <mRMR> <nslkddfolder>"
    )
    sys.exit()
else:
    pytime = Path(sys.argv[1])
    IR_time = Path(sys.argv[2])
    LHR_time = Path(sys.argv[3])
    mRMR_time = Path(sys.argv[4])
    nslkddfolder = Path(sys.argv[5])

with open(pytime, "rb") as handle:
    pytimes = pickle.load(handle)

IRlf = pd.read_csv(IR_time, header=None, index_col=0)
LHRlf = pd.read_csv(LHR_time, header=None, index_col=0)
mRMR = pd.read_csv(mRMR_time, header=None, index_col=0)

nslkdd_times = pd.read_csv(
    nslkddfolder.joinpath("elapsed_times.csv"), index_col=0
)
nslkdd_times.columns = ["NSL-KDD"]
nslkdd_IRlf = float(
    pd.read_csv(nslkddfolder.joinpath("IRelief/tI.csv")).columns[0]
)
nslkdd_LHRlf = float(
    pd.read_csv(nslkddfolder.joinpath("LHRelief/tLH.csv")).columns[0]
)
nslkdd_mRMR = float(
    pd.read_csv(nslkddfolder.joinpath("mRMR/tmRMR.csv")).columns[0]
)
nslkdd_times.loc["IRlf", "NSL-KDD"] = nslkdd_IRlf
nslkdd_times.loc["LHRlf", "NSL-KDD"] = nslkdd_LHRlf
nslkdd_times.loc["mRMR", "NSL-KDD"] = nslkdd_mRMR

FSS = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini",  "MI",
    "mRMR", "FT",
    "PDE-S"
]

datasets = [
    "cns", "lung", "leuk",
    "colon", "pros3", "gcm",
    "geneExpressionCancerRNA", "PersonGaitDataSet"
]

elapsedTimes_df = pd.DataFrame(
    data=np.zeros((len(datasets), len(FSS))),
    index=datasets, columns=FSS
)

for fss in FSS:
    for ds in datasets:
        if fss == "IRlf":
            elapsedTimes_df.at[ds, fss] = IRlf.at[ds, 1]
        elif fss == "LHRlf":
            elapsedTimes_df.at[ds, fss] = LHRlf.at[ds, 1]
        elif fss == "mRMR":
            elapsedTimes_df.at[ds, fss] = mRMR.at[ds, 1]
        else:
            elapsedTimes_df.at[ds, fss] = pytimes[ds][fss]

elapsedTimes_df = pd.concat(
    [elapsedTimes_df, (nslkdd_times.T)[FSS]]
)
print(elapsedTimes_df.round(decimals=3))
print(elapsedTimes_df.mean())
print("Average of all average computational times:")
print((elapsedTimes_df.mean()).mean())
print(elapsedTimes_df.to_latex(index=False, float_format="%.3f"))

sys.exit(0)
