import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 3:
    print("Possible usage: python3.11 <resultsfolder> <datasetname>")
    sys.exit(1)
else:
    resultsfolder = Path(sys.argv[1])
    datasetname = sys.argv[2]

clf_dict = {
    "kNN": "kNN", "SVM": "SVM", "NB": "GNB", "LDA": "LDA", "DT": "DT"
}

averaged10foldcv = pd.read_csv(
    resultsfolder.joinpath(f"{datasetname}10foldcv_averaged.csv")
)
n30 = averaged10foldcv[averaged10foldcv["nObs"]==30.0]
n50 = averaged10foldcv[averaged10foldcv["nObs"]==50.0]
n70 = averaged10foldcv[averaged10foldcv["nObs"]==70.0]

sucrates = pd.read_csv(
    resultsfolder.joinpath(f"{datasetname}_successrates.csv"), index_col=0
)
sucrates_n30 = sucrates.loc[30]
sucrates_n50 = sucrates.loc[50]
sucrates_n70 = sucrates.loc[70]

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S"
]
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

final_df.to_excel(resultsfolder.joinpath(f"{datasetname}_latexTable.xlsx"))

sys.exit(0)

