import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

if len(sys.argv) < 5:
    print("Possible usage: python3 readElapsedTime.py <pyTime> <IRlf> <LHRlf> <mRMR>")
    sys.exit()
else:
    pytime = Path(sys.argv[1])
    IR_time = Path(sys.argv[2])
    LHR_time = Path(sys.argv[3])
    mRMR_time = Path(sys.argv[4])

with open(pytime, "rb") as handle:
    pytimes = pickle.load(handle)

IRlf = pd.read_csv(IR_time, header=None, index_col=0)
LHRlf = pd.read_csv(LHR_time, header=None, index_col=0)
mRMR = pd.read_csv(mRMR_time, header=None, index_col=0)

FSS = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini",  "MI",
    "mRMR", "FT",
    "OAtotal", "OApw"
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

elapsedTimes_df = elapsedTimes_df.rename(
    columns={"OAtotal": "PDE-S", "OApw": "PDE-S*"}
)
print(elapsedTimes_df.to_csv())
# elapsedTimes_df = elapsedTimes_df.round(decimals=3)
# print(elapsedTimes_df)
# print(elapsedTimes_df.to_latex(index=False, float_format="%.2f"))

sys.exit(0)
