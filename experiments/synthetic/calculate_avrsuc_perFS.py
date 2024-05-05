import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

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
    "OA": "PDE-S",
    "OApw": "PDE-S*"
}
CLF_dict = {
    "kNN": "kNN", "SVM": "SVM", "NB": "GNB", "LDA": "LDA", "DT": "DT"
}

if len(sys.argv) < 2:
    print("Possible usage: python3.11 calculate_avrsuc_perFS.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

electricalFolder = folder.joinpath("Electrical/Results")

# === === === ===
# ANDOR discrete
ANDORdiscrete = pd.read_csv(
    electricalFolder.joinpath("ANDORdiscrete_successrates.csv"),
    index_col=0
)
ANDORdiscrete_10foldcv = pd.read_csv(
    electricalFolder.joinpath("ANDORdiscrete10foldcv_averaged.csv")
)[["FS", "Clf", "Bal.Acc"]].groupby(["FS", "Clf"]).mean()

# ANDOR continuous
ANDORcontinuous = pd.read_csv(
    electricalFolder.joinpath("ANDORcontinuous_successrates.csv"),
    index_col=0
)
ANDORcontinuous_10foldcv = pd.read_csv(
    electricalFolder.joinpath("ANDORcontinuous10foldcv_averaged.csv")
)[["FS", "Clf", "Bal.Acc"]].groupby(["FS", "Clf"]).mean()

# === === === ===
# ADDER discrete
ADDERdiscrete = pd.read_csv(
    electricalFolder.joinpath("ADDERdiscrete_successrates.csv"),
    index_col=0
)
ADDERdiscrete_10foldcv = pd.read_csv(
    electricalFolder.joinpath("ADDERdiscrete10foldcv_averaged.csv")
)[["FS", "Clf", "Bal.Acc"]].groupby(["FS", "Clf"]).mean()

# ADDER continuous
ADDERcontinuous = pd.read_csv(
    electricalFolder.joinpath("ADDERcontinuous_successrates.csv"),
    index_col=0
)
ADDERcontinuous_10foldcv = pd.read_csv(
    electricalFolder.joinpath("ADDERcontinuous10foldcv_averaged.csv")
)[["FS", "Clf", "Bal.Acc"]].groupby(["FS", "Clf"]).mean()

# === === === ===
# SD
SDFolder = folder.joinpath("SDI/Results")
SDI = pd.read_csv(SDFolder.joinpath("20_SDIsuccessrates.csv"), index_col=0)
SDI_10foldcv = pd.read_csv(
    SDFolder.joinpath("10foldcv_averaged.csv")
)[["FS", "Clf", "Bal.Acc"]].groupby(["FS", "Clf"]).mean()

avr_df = pd.DataFrame(
    data=np.zeros((9, 6)),
    index=["RlfF", "MSurf", "IRlf", "LHRlf", "RFGini", "MI", "mRMR", "FT", "OA"],
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
print(avr_df)

avr_df_str = avr_df.to_latex(float_format="%.3f")
print(avr_df_str)

sys.exit(0)
