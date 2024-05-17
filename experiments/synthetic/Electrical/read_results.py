import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 3:
    print("Possible usage: python3.11 <folder> <datasetname>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])
    datasetname = sys.argv[2]

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S"
]

averaged10foldcv = pd.read_csv(
    folder.joinpath(f"OtherFS/{datasetname}10foldcv_averaged.csv")
)
full10foldcv = pd.read_csv(
    folder.joinpath(f"OtherFS/{datasetname}10foldcv.csv"), index_col=0
)
pdes_averaged10foldcv = pd.read_csv(
    folder.joinpath(f"PDE-S/{datasetname}10foldcvPDE-S_averaged.csv")
)
pdes_full10foldcv = pd.read_csv(
    folder.joinpath(f"PDE-S/{datasetname}10foldcvPDE-S.csv"), index_col=0
)
averaged10foldcv = pd.concat([averaged10foldcv, pdes_averaged10foldcv])
averaged10foldcv = averaged10foldcv.reset_index(drop=True)


# === === === ===
# Computing the sd of the balanced accuracies across all iterations
full10foldcv = pd.concat([full10foldcv, pdes_full10foldcv])
full10foldcv_n30 = full10foldcv[full10foldcv.nObs == 30.0].reset_index(drop=True)
full10foldcv_n30 = full10foldcv_n30[["FS", "Clf", "Bal.Acc"]]
full10foldcv_n50 = full10foldcv[full10foldcv.nObs == 50.0].reset_index(drop=True)
full10foldcv_n50 = full10foldcv_n50[["FS", "Clf", "Bal.Acc"]]
full10foldcv_n70 = full10foldcv[full10foldcv.nObs == 70.0].reset_index(drop=True)
full10foldcv_n70 = full10foldcv_n70[["FS", "Clf", "Bal.Acc"]]

full10foldcv_n30_sd = full10foldcv_n30.groupby(["FS", "Clf"]).std()
full10foldcv_n50_sd = full10foldcv_n50.groupby(["FS", "Clf"]).std()
full10foldcv_n70_sd = full10foldcv_n70.groupby(["FS", "Clf"]).std()

def fill_df_sd(_dfsd):
    table_df = pd.DataFrame(
        data=np.zeros((len(fsorder), 5)), index=fsorder,
        columns=["kNN", "SVM", "NB", "LDA", "DT"]
    )
    for idxs in _dfsd.index:
        fs_i = idxs[0]; clf_i = idxs[1]
        table_df.at[fs_i, clf_i] = _dfsd.loc[idxs].values[0]

    return table_df

n30_sd = fill_df_sd(full10foldcv_n30_sd)
n50_sd = fill_df_sd(full10foldcv_n50_sd)
n70_sd = fill_df_sd(full10foldcv_n70_sd)


# === === === ===
# Getting the sd of the success rates across all iterations


# === === === ===
# Getting all averaged balanced accuracies and success rates
clf_dict = {
    "kNN": "kNN", "SVM": "SVM", "NB": "GNB", "LDA": "LDA", "DT": "DT"
}
n30 = averaged10foldcv[averaged10foldcv["nObs"]==30.0]
n50 = averaged10foldcv[averaged10foldcv["nObs"]==50.0]
n70 = averaged10foldcv[averaged10foldcv["nObs"]==70.0]

sucrates = pd.read_csv(
    folder.joinpath(f"Suc/{datasetname}_successrates.csv"), index_col=0
)
sucrates_n30 = sucrates.loc[30]
sucrates_n50 = sucrates.loc[50]
sucrates_n70 = sucrates.loc[70]

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

n30_df = fill_df(n30, sucrates_n30)
n50_df = fill_df(n50, sucrates_n50)
n70_df = fill_df(n70, sucrates_n70)

final_df = pd.concat([n30_df, n50_df, n70_df])
print(final_df)

# final_df.to_excel(folder.joinpath(f"{datasetname}_latexTable.xlsx"))

sys.exit(0)

