import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3.11 read_results.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S"
]

averaged10foldcv = pd.read_csv(
    folder.joinpath(f"OtherFS/10foldcv_averaged.csv")
)
full10foldcv = pd.read_csv(
    folder.joinpath(f"OtherFS/10foldcv.csv"), index_col=0
)
pdes_averaged10foldcv = pd.read_csv(
    folder.joinpath(f"PDE-S/10foldcvPDE-S_averaged.csv")
)
pdes_full10foldcv = pd.read_csv(
    folder.joinpath(f"PDE-S/10foldcvPDE-S.csv"), index_col=0
)
averaged10foldcv = pd.concat([averaged10foldcv, pdes_averaged10foldcv])
averaged10foldcv = averaged10foldcv.reset_index(drop=True)


# === === === ===
# Computing the sd of the balanced accuracies across all iterations
full10foldcv = pd.concat([full10foldcv, pdes_full10foldcv])
full10foldcv_class2 = full10foldcv[full10foldcv.nClass == 2.0].reset_index(drop=True)
full10foldcv_class2 = full10foldcv_class2[["FS", "Clf", "Bal.Acc"]]
full10foldcv_class3 = full10foldcv[full10foldcv.nClass == 3.0].reset_index(drop=True)
full10foldcv_class3 = full10foldcv_class3[["FS", "Clf", "Bal.Acc"]]
full10foldcv_class4 = full10foldcv[full10foldcv.nClass == 4.0].reset_index(drop=True)
full10foldcv_class4 = full10foldcv_class4[["FS", "Clf", "Bal.Acc"]]

full10foldcv_class2_sd = full10foldcv_class2.groupby(["FS", "Clf"]).std()
full10foldcv_class3_sd = full10foldcv_class3.groupby(["FS", "Clf"]).std()
full10foldcv_class4_sd = full10foldcv_class4.groupby(["FS", "Clf"]).std()

def fill_df_sd(_dfsd):
    table_df = pd.DataFrame(
        data=np.zeros((len(fsorder), 5)), index=fsorder,
        columns=["kNN", "SVM", "NB", "LDA", "DT"]
    )
    for idxs in _dfsd.index:
        fs_i = idxs[0]; clf_i = idxs[1]
        table_df.at[fs_i, clf_i] = _dfsd.loc[idxs].values[0]

    return table_df

class2_sd = fill_df_sd(full10foldcv_class2_sd)
class3_sd = fill_df_sd(full10foldcv_class3_sd)
class4_sd = fill_df_sd(full10foldcv_class4_sd)


# === === === ===
# Getting all averaged balanced accuracies and success rates
clf_dict = {
    "kNN": "kNN", "SVM": "SVM", "NB": "GNB", "LDA": "LDA", "DT": "DT"
}
class2 = averaged10foldcv[averaged10foldcv["nClass"]==2]
class3 = averaged10foldcv[averaged10foldcv["nClass"]==3]
class4 = averaged10foldcv[averaged10foldcv["nClass"]==4]

sucrates = pd.read_csv(
    folder.joinpath("Suc/20_SDIsuccessrates.csv"), index_col=0
)
sucrates_class2 = sucrates.loc[2.0]
sucrates_class3 = sucrates.loc[3.0]
sucrates_class4 = sucrates.loc[4.0]

sucrates_sd = pd.read_csv(
    folder.joinpath(f"Suc/20_SDIsuccessrates_sd.csv"), index_col=0
)
sucrates_class2_sd = sucrates_sd.loc[2.0]
sucrates_class3_sd = sucrates_sd.loc[3.0]
sucrates_class4_sd = sucrates_sd.loc[4.0]

def fill_df(_df, _sucrates):
    table_df = pd.DataFrame(
        data=np.zeros((len(fsorder), 5)), index=fsorder,
        columns=["kNN", "SVM", "GNB", "LDA", "DT"]
    )
    for i in _df.index:
        i_fss = _df.at[i, "FS"]
        i_clf = _df.at[i, "Clf"]
        table_df.at[i_fss, clf_dict[i_clf]] = _df.at[i, "Bal.Acc"]

    _sucrates = _sucrates.loc[fsorder]
    table_df["Suc."] = _sucrates
    table_df = table_df[
        ["Suc.", "kNN", "SVM", "GNB", "LDA", "DT"]
    ]
    table_df.loc["Average"] = table_df.mean()

    return table_df.round(decimals=3)

class2_df = fill_df(class2, sucrates_class2)
class3_df = fill_df(class3, sucrates_class3)
class4_df = fill_df(class4, sucrates_class4)

def puttogether_df_sd(_clfsd, _sucsd):
    _clfsd["Suc."] = _sucsd
    _clfsd = _clfsd[["Suc.", "kNN", "SVM", "NB", "LDA", "DT"]]
    _clfsd.loc["Average"] = _clfsd.mean()
    return _clfsd.round(decimals=2)

class2_df_sd = puttogether_df_sd(class2_sd, sucrates_class2_sd)
class3_df_sd = puttogether_df_sd(class3_sd, sucrates_class3_sd)
class4_df_sd = puttogether_df_sd(class4_sd, sucrates_class4_sd)

final_df = pd.concat([class2_df, class3_df, class4_df])
print(final_df)

final_df_sd = pd.concat([class2_df_sd, class3_df_sd, class4_df_sd])
print(final_df_sd)

final_df.to_excel("SDI_latexTable.xlsx")
final_df_sd.to_excel("SDI_sd_latexTable.xlsx")

sys.exit(0)

