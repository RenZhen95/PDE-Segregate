import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 5:
    print("Possible usage: python3 plotElapsedTime.py <pyTime> <IRlf> <LHRlf> <savefile>")
    sys.exit()
else:
    pytime = Path(sys.argv[1])
    IR_time = Path(sys.argv[2])
    LHR_time = Path(sys.argv[3])
    csvfile = sys.argv[4]

with open(pytime, "rb") as handle:
    pytimes = pickle.load(handle)

IRlf = pd.read_csv(IR_time, header=None, index_col=0)
LHRlf = pd.read_csv(LHR_time, header=None, index_col=0)

FSS = [
    "RlfF", "MSurf", "I-RELIEF", "LH-RELIEF",
    "MI",
    "RFGini",
    "FT",
    "OAtotal", "OApw"
]

datasets = [
    "cns", "lung", "leuk",
    "dlbcl", "pros3", "geneExpressionCancerRNA",
    "colon", "gcm"
]

elapsedTimes_df = pd.DataFrame(
    data=np.zeros((len(datasets), len(FSS))),
    index=datasets, columns=FSS
)

for fss in FSS:
    for ds in datasets:
        if fss == "I-RELIEF":
            elapsedTimes_df.at[ds, fss] = IRlf.at[ds, 1]
        elif fss == "LH-RELIEF":
            elapsedTimes_df.at[ds, fss] = LHRlf.at[ds, 1]
        else:
            elapsedTimes_df.at[ds, fss] = pytimes[ds][fss]

averageElapsedTime = elapsedTimes_df.mean(axis=0)

averagedTimes_row = pd.DataFrame(
    data=np.reshape(averageElapsedTime.values, (1, elapsedTimes_df.shape[1])),
    columns=averageElapsedTime.index, index=["Average"]
)

elapsedTimes_df = elapsedTimes_df._append(averagedTimes_row)

elapsedTimes_df.to_csv(f"{csvfile}.csv")

sys.exit()


    
