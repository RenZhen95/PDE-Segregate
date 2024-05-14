import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

if len(sys.argv) < 2:
    print(
        "Possible usage: python3.11 evaluate_elapsedtime.py <resultsFolder>"
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])

# Getting elapsed times (SDI)
SDIfolder = resultsFolder.joinpath("SDI/Results_052024/Combined")
SDI_df = pd.read_csv(
    SDIfolder.joinpath("SDIelapsedtimes.csv"), index_col=0
)
SDIaverage_df = SDI_df.groupby(["nClass"]).mean()
SDIaverage_df = SDIaverage_df.drop(columns=["iteration"])
SDIaverage_df.rename(
    index={2.0: "SD-2", 3.0: "SD-3", 4.0: "SD-4"}, inplace=True
)

# Getting elapsed times (Electrical)
Electricalfolder = resultsFolder.joinpath("Electrical/Results_052024/Combined")
ANDORdiscrete_df = pd.read_csv(
    Electricalfolder.joinpath(f"ANDORdiscrete_elapsedtimes.csv"), index_col=0
)
ANDORdiscreteaverage_df = ANDORdiscrete_df.groupby(["n_obs"]).mean()
ANDORdiscreteaverage_df = ANDORdiscreteaverage_df.drop(columns=["iteration"])
ANDORdiscreteaverage_df.rename(
    index={
        30.0: "Disc. ANDOR (n30)",
        50.0: "Disc. ANDOR (n50)",
        70.0: "Disc. ANDOR (n70)"
    }, inplace=True
)
ANDORcontinuous_df = pd.read_csv(
    Electricalfolder.joinpath(f"ANDORcontinuous_elapsedtimes.csv"), index_col=0
)
ANDORcontinuousaverage_df = ANDORcontinuous_df.groupby(["n_obs"]).mean()
ANDORcontinuousaverage_df = ANDORcontinuousaverage_df.drop(columns=["iteration"])
ANDORcontinuousaverage_df.rename(
    index={
        30.0: "Cont. ANDOR (n30)",
        50.0: "Cont. ANDOR (n50)",
        70.0: "Cont. ANDOR (n70)"
    }, inplace=True
)
# print(ANDORcontinuousaverage_df)

ADDERdiscrete_df = pd.read_csv(
    Electricalfolder.joinpath(f"ADDERdiscrete_elapsedtimes.csv"), index_col=0
)
ADDERdiscreteaverage_df = ADDERdiscrete_df.groupby(["n_obs"]).mean()
ADDERdiscreteaverage_df = ADDERdiscreteaverage_df.drop(columns=["iteration"])
ADDERdiscreteaverage_df.rename(
    index={
        30.0: "Disc. ADDER (n30)",
        50.0: "Disc. ADDER (n50)",
        70.0: "Disc. ADDER (n70)"
    }, inplace=True
)
# print(ADDERdiscreteaverage_df)

ADDERcontinuous_df = pd.read_csv(
    Electricalfolder.joinpath(f"ADDERcontinuous_elapsedtimes.csv"), index_col=0
)
ADDERcontinuousaverage_df = ADDERcontinuous_df.groupby(["n_obs"]).mean()
ADDERcontinuousaverage_df = ADDERcontinuousaverage_df.drop(columns=["iteration"])
ADDERcontinuousaverage_df.rename(
    index={
        30.0: "Cont. ADDER (n30)",
        50.0: "Cont. ADDER (n50)",
        70.0: "Cont. ADDER (n70)"
    }, inplace=True
)
# print(ADDERcontinuousaverage_df)

average_df = pd.concat(
    [
        ANDORdiscreteaverage_df, ANDORcontinuousaverage_df,
        ADDERdiscreteaverage_df, ADDERcontinuousaverage_df,
        SDIaverage_df
    ]
)
average_df.loc["Average"] = average_df.mean()

average_df = average_df[
    [
        "RlfF", "MSurf", "IRlf", "LHRlf",
        "RFGini", "MI", "mRMR", "FT", "PDE-S"
    ]
]
print(average_df.round(decimals=2))
avrOfAverages = average_df.loc['Average',:].mean()
print(f"Average of averages: {round(avrOfAverages,2)}")
print(average_df.to_latex(float_format="%.3f"))

sys.exit(0)
