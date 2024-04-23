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
SDIfolder = resultsFolder.joinpath("SDI/Combined")
SDI_df = pd.read_csv(
    SDIfolder.joinpath("SDIelapsedtimes.csv"), index_col=0
)
SDIaverage_df = SDI_df.groupby(["nClass"]).mean()
SDIaverage_df = SDIaverage_df.drop(columns=["iteration"])
SDIaverage_df.rename(
    index={2.0: "SD-2", 3.0: "SD-3", 4.0: "SD-4"}, inplace=True
)
# print(SDIaverage_df)

# Getting elapsed times (Electrical)
Electricalfolder = resultsFolder.joinpath("Electrical/Combined")
ANDORdiscrete_df = pd.read_csv(
    Electricalfolder.joinpath(f"ANDORdiscrete_elapsedtimes.csv"), index_col=0
)
ANDORdiscreteaverage_df = ANDORdiscrete_df.groupby(["n_obs"]).mean()
ANDORdiscreteaverage_df = ANDORdiscreteaverage_df.drop(columns=["iteration"])
ANDORdiscreteaverage_df.rename(
    index={
        30.0: "Discrete ANDOR (n30)",
        50.0: "Discrete ANDOR (n50)",
        70.0: "Discrete ANDOR (n70)"
    }, inplace=True
)
# print(ANDORdiscreteaverage_df)

ANDORcontinuous_df = pd.read_csv(
    Electricalfolder.joinpath(f"ANDORcontinuous_elapsedtimes.csv"), index_col=0
)
ANDORcontinuousaverage_df = ANDORcontinuous_df.groupby(["n_obs"]).mean()
ANDORcontinuousaverage_df = ANDORcontinuousaverage_df.drop(columns=["iteration"])
ANDORcontinuousaverage_df.rename(
    index={
        30.0: "Continuous ANDOR (n30)",
        50.0: "Continuous ANDOR (n50)",
        70.0: "Continuous ANDOR (n70)"
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
        30.0: "Discrete ADDER (n30)",
        50.0: "Discrete ADDER (n50)",
        70.0: "Discrete ADDER (n70)"
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
        30.0: "Continuous ADDER (n30)",
        50.0: "Continuous ADDER (n50)",
        70.0: "Continuous ADDER (n70)"
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

average_df.rename(
    columns={
        "RlfF": "ReliefF", "MSurf": "MultiSURF",
        "IRlf": "I-Relief", "LHRlf": "LH-Relief",
        "RFGini": "RF (Gini)", "FT": "F-Ratio",
        "OA": "PDE-S", "OApw": "PDE-S*"
    }, inplace=True
)
average_df = average_df[
    [
        "ReliefF", "MultiSURF", "I-Relief", "LH-Relief",
        "RF (Gini)", "MI", "mRMR", "F-Ratio",
        "PDE-S", "PDE-S*"
    ]
]
print(average_df)
average_df = average_df.round(decimals=3)

# average_df.to_csv("average_elapsedtime.csv")

sys.exit(0)
