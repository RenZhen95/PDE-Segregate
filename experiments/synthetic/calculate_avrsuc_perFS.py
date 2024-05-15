import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

def read_10foldcvresults_electrical(_folder, _dataset):
    averaged10foldcv = pd.read_csv(
        _folder.joinpath(f"OtherFS/{_dataset}10foldcv_averaged.csv")
    )
    pdes_averaged10foldcv = pd.read_csv(
        _folder.joinpath(f"PDE-S/{_dataset}10foldcvPDE-S_averaged.csv")
    )
    averaged10foldcv = pd.concat([averaged10foldcv, pdes_averaged10foldcv])
    averaged10foldcv = averaged10foldcv.reset_index(drop=True)
    averaged10foldcv = averaged10foldcv[
        ["FS", "Clf", "Bal.Acc"]
    ].groupby(["FS", "Clf"]).mean()

    return averaged10foldcv

# FSS name-mapper
FSS_dict = {
    "RlfF": "RlfF",
    "MSurf": "MSurf",
    "RlfI": "IRlf",
    "RlfLM": "LHRlf",
    "RFGini": "RFGini",
    "MI": "MI",
    "FT": "FT",
    "mRMR": "mRMR",
    "PDE-S": "PDE-S"
}
CLF_dict = {
    "kNN": "kNN", "SVM": "SVM", "NB": "GNB", "LDA": "LDA", "DT": "DT"
}

if len(sys.argv) < 2:
    print("Possible usage: python3.11 calculate_avrsuc_perFS.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

electricalFolder = folder.joinpath("Electrical/Results_052024_1p5")

# === === === ===
# ANDOR discrete
ANDORdiscrete = pd.read_csv(
    electricalFolder.joinpath("Suc/ANDORdiscrete_successrates.csv"),
    index_col=0
)
ANDORdiscrete_10foldcv = read_10foldcvresults_electrical(
    electricalFolder, "ANDORdiscrete"
)

# ANDOR continuous
ANDORcontinuous = pd.read_csv(
    electricalFolder.joinpath("Suc/ANDORcontinuous_successrates.csv"),
    index_col=0
)
ANDORcontinuous_10foldcv = read_10foldcvresults_electrical(
    electricalFolder, "ANDORcontinuous"
)

# === === === ===
# ADDER discrete
ADDERdiscrete = pd.read_csv(
    electricalFolder.joinpath("Suc/ADDERdiscrete_successrates.csv"),
    index_col=0
)
ADDERdiscrete_10foldcv = read_10foldcvresults_electrical(
    electricalFolder, "ADDERdiscrete"
)

# ADDER continuous
ADDERcontinuous = pd.read_csv(
    electricalFolder.joinpath("Suc/ADDERcontinuous_successrates.csv"),
    index_col=0
)
ADDERcontinuous_10foldcv = read_10foldcvresults_electrical(
    electricalFolder, "ANDORcontinuous"
)

# === === === ===
# SD
SDFolder = folder.joinpath("SDI/Results_052024_1p5")
SDI = pd.read_csv(SDFolder.joinpath("Suc/20_SDIsuccessrates.csv"), index_col=0)
SDIaveraged10foldcv = pd.read_csv(
    SDFolder.joinpath(f"OtherFS/10foldcv_averaged.csv")
)
SDIpdes_averaged10foldcv = pd.read_csv(
    SDFolder.joinpath(f"PDE-S/10foldcvPDE-S_averaged.csv")
)
SDI_10foldcv = pd.concat([SDIaveraged10foldcv, SDIpdes_averaged10foldcv])
SDI_10foldcv = SDI_10foldcv.reset_index(drop=True)
SDI_10foldcv = SDI_10foldcv[
    ["FS", "Clf", "Bal.Acc"]
].groupby(["FS", "Clf"]).mean()

avr_df = pd.DataFrame(
    data=np.zeros((9, 6)),
    index=["RlfF", "MSurf", "IRlf", "LHRlf", "RFGini", "MI", "mRMR", "FT", "PDE-S"],
    columns=["Suc.", "kNN", "SVM", "NB", "LDA", "DT"]
)

success_rates = pd.concat(
    [
        ANDORdiscrete, ANDORcontinuous,
        ADDERdiscrete, ADDERcontinuous,
        SDI
    ], ignore_index=True
)

avr_df["Suc."] = success_rates.mean()

avr_10foldcv = pd.concat(
    [
        ANDORdiscrete_10foldcv,
        ANDORcontinuous_10foldcv,
        ADDERdiscrete_10foldcv,
        ADDERcontinuous_10foldcv,
        SDI_10foldcv
    ], axis=1
).mean(axis=1)

avr_df["kNN"] = avr_10foldcv.xs("kNN", level=1)
avr_df["SVM"] = avr_10foldcv.xs("SVM", level=1)
avr_df["NB"] = avr_10foldcv.xs("NB", level=1)
avr_df["LDA"] = avr_10foldcv.xs("LDA", level=1)
avr_df["DT"] = avr_10foldcv.xs("DT", level=1)

avr_df.rename(columns=CLF_dict, inplace=True)
avr_df.rename(index=FSS_dict, inplace=True)
avr_df.loc["Average",:] = avr_df.mean()
avr_df = avr_df.round(decimals=3)
print(avr_df)
print(avr_df.to_csv())

# avr_df_str = avr_df.to_latex(float_format="%.3f")
# print(avr_df_str)

sys.exit(0)
