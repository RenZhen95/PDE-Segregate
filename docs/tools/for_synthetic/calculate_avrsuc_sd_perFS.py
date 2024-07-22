import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

def read_10foldcvresults_electrical(_folder, _dataset):
    full10foldcv = pd.read_csv(
        _folder.joinpath(f"OtherFS/{_dataset}10foldcv.csv"), index_col=0
    )
    pdes_full10foldcv = pd.read_csv(
        _folder.joinpath(f"PDE-S/{_dataset}10foldcvPDE-S.csv"), index_col=0
    )
    full10foldcv = pd.concat([full10foldcv, pdes_full10foldcv])
    full10foldcv = full10foldcv.reset_index(drop=True)

    full10foldcv_n30 = full10foldcv[full10foldcv.nObs == 30.0].reset_index(drop=True)
    full10foldcv_n30 = full10foldcv_n30[["FS", "Clf", "Bal.Acc"]]
    full10foldcv_n30_sd = full10foldcv_n30.groupby(["FS", "Clf"]).std()
    full10foldcv_n30_sd.columns = ["n30_sd"]

    full10foldcv_n50 = full10foldcv[full10foldcv.nObs == 50.0].reset_index(drop=True)
    full10foldcv_n50 = full10foldcv_n50[["FS", "Clf", "Bal.Acc"]]
    full10foldcv_n50_sd = full10foldcv_n50.groupby(["FS", "Clf"]).std()
    full10foldcv_n50_sd.columns = ["n50_sd"]

    full10foldcv_n70 = full10foldcv[full10foldcv.nObs == 70.0].reset_index(drop=True)
    full10foldcv_n70 = full10foldcv_n70[["FS", "Clf", "Bal.Acc"]]
    full10foldcv_n70_sd = full10foldcv_n70.groupby(["FS", "Clf"]).std()
    full10foldcv_n70_sd.columns = ["n70_sd"]

    full10foldcv_sd = pd.concat(
        [full10foldcv_n30_sd, full10foldcv_n50_sd, full10foldcv_n70_sd], axis=1
    )

    return full10foldcv_sd.mean(axis=1)

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
    print("Possible usage: python3.11 calculate_avrsuc_sd_perFS.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

electricalFolder = folder.joinpath("Electrical/Results_052024_1p5")

# === === === ===
# ANDOR discrete
ANDORdiscrete = pd.read_csv(
    electricalFolder.joinpath("Suc/ANDORdiscrete_successrates_sd.csv"),
    index_col=0
)
ANDORdiscrete_10foldcv_sd = read_10foldcvresults_electrical(
    electricalFolder, "ANDORdiscrete"
)

# ANDOR continuous
ANDORcontinuous = pd.read_csv(
    electricalFolder.joinpath("Suc/ANDORcontinuous_successrates_sd.csv"),
    index_col=0
)
ANDORcontinuous_10foldcv_sd = read_10foldcvresults_electrical(
    electricalFolder, "ANDORcontinuous"
)

# === === === ===
# ADDER discrete
ADDERdiscrete = pd.read_csv(
    electricalFolder.joinpath("Suc/ADDERdiscrete_successrates_sd.csv"),
    index_col=0
)
ADDERdiscrete_10foldcv_sd = read_10foldcvresults_electrical(
    electricalFolder, "ADDERdiscrete"
)

# ADDER continuous
ADDERcontinuous = pd.read_csv(
    electricalFolder.joinpath("Suc/ADDERcontinuous_successrates_sd.csv"),
    index_col=0
)
ADDERcontinuous_10foldcv_sd = read_10foldcvresults_electrical(
    electricalFolder, "ANDORcontinuous"
)

# === === === ===
# SD
SDFolder = folder.joinpath("SDI/Results_052024_1p5")
SDI = pd.read_csv(SDFolder.joinpath("Suc/20_SDIsuccessrates.csv"), index_col=0)

SDIfull10foldcv = pd.read_csv(
    SDFolder.joinpath(f"OtherFS/10foldcv.csv"), index_col=0
)
SDIpdes_full10foldcv = pd.read_csv(
    SDFolder.joinpath(f"PDE-S/10foldcvPDE-S.csv"), index_col=0
)
SDI_10foldcv = pd.concat([SDIfull10foldcv, SDIpdes_full10foldcv])

SDI_10foldcv_class2 = SDI_10foldcv[SDI_10foldcv.nClass == 2.0].reset_index(drop=True)
SDI_10foldcv_class2 = SDI_10foldcv_class2[["FS", "Clf", "Bal.Acc"]]
SDI_10foldcv_class2_sd = SDI_10foldcv_class2.groupby(["FS", "Clf"]).std()
SDI_10foldcv_class2_sd.columns = ["class2_sd"]

SDI_10foldcv_class3 = SDI_10foldcv[SDI_10foldcv.nClass == 3.0].reset_index(drop=True)
SDI_10foldcv_class3 = SDI_10foldcv_class3[["FS", "Clf", "Bal.Acc"]]
SDI_10foldcv_class3_sd = SDI_10foldcv_class3.groupby(["FS", "Clf"]).std()
SDI_10foldcv_class3_sd.columns = ["class3_sd"]

SDI_10foldcv_class4 = SDI_10foldcv[SDI_10foldcv.nClass == 4.0].reset_index(drop=True)
SDI_10foldcv_class4 = SDI_10foldcv_class4[["FS", "Clf", "Bal.Acc"]]
SDI_10foldcv_class4_sd = SDI_10foldcv_class4.groupby(["FS", "Clf"]).std()
SDI_10foldcv_class4_sd.columns = ["class4_sd"]

SDI_10foldcv_sd = pd.concat(
    [SDI_10foldcv_class2_sd, SDI_10foldcv_class3_sd, SDI_10foldcv_class4_sd], axis=1
)
SDI_10foldcv_sd = SDI_10foldcv_sd.mean(axis=1)

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

avr_10foldcv_sd = pd.concat(
    [
        ANDORdiscrete_10foldcv_sd,
        ANDORcontinuous_10foldcv_sd,
        ADDERdiscrete_10foldcv_sd,
        ADDERcontinuous_10foldcv_sd,
        SDI_10foldcv_sd
    ], axis=1
).mean(axis=1)

avr_df["kNN"] = avr_10foldcv_sd.xs("kNN", level=1)
avr_df["SVM"] = avr_10foldcv_sd.xs("SVM", level=1)
avr_df["NB"] = avr_10foldcv_sd.xs("NB", level=1)
avr_df["LDA"] = avr_10foldcv_sd.xs("LDA", level=1)
avr_df["DT"] = avr_10foldcv_sd.xs("DT", level=1)

avr_df.rename(columns=CLF_dict, inplace=True)
avr_df.rename(index=FSS_dict, inplace=True)
avr_df.loc["Average",:] = avr_df.mean()
avr_df = avr_df.round(decimals=3)
print(avr_df)
print(avr_df.to_csv(sep=' '))

# # avr_df_str = avr_df.to_latex(float_format="%.3f")
# # print(avr_df_str)

sys.exit(0)
