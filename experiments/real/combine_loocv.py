import os, sys
import pickle
import pandas as pd
from pathlib import Path

if len(sys.argv) < 3:
    print("Possible usage: python3.11 combine_pyFS.py <otherfsfolder> <pdesfolder>")
    print(
        "Script to combine the loocv and computational time from FS methods " +
        "implemented in Python"
    )
    sys.exit(1)
else:
    otherfs = Path(sys.argv[1])
    pdes = Path(sys.argv[2])

otherfs_loocv = otherfs.joinpath("loocv")
pdes_loocv = pdes.joinpath("loocv")

for f in os.scandir(otherfs_loocv):
    datasetname = (f.name).split('_')[0]

    print(f"Combining the LOOCV for the datasets {datasetname} ... ")

    df_otherfs = pd.read_csv(f, index_col=0)
    df_pdes = pd.read_csv(
        pdes_loocv.joinpath(f"{datasetname}PDE-S_LOOCV.csv"), index_col=0
    )

    combined_df = pd.concat([df_otherfs, df_pdes])

    combined_df.to_csv(f"{datasetname}_LOOCV.csv")

with open(otherfs.joinpath("fsElapsedTimes.pkl"), "rb") as handle:
    otherfs_time = pickle.load(handle)

with open(pdes.joinpath("fsElapsedTimes_PDE-S.pkl"), "rb") as handle:
    pdes_time = pickle.load(handle)

for ds in otherfs_time.keys():
    otherfs_time[ds].update({"PDE-S": pdes_time[ds]})

with open("fsElapsedTimes.pkl", "wb") as handle:
    pickle.dump(otherfs_time, handle)

sys.exit(0)
