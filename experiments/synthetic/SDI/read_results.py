import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3.11 read_results.py <resultsfolder>")
    sys.exit(1)
else:
    resultsfolder = Path(sys.argv[1])

clf_dict = {
    "kNN": "kNN", "SVM": "SVM", "NB": "GNB", "LDA": "LDA", "DT": "DT"
}

averaged10foldcv = pd.read_csv(
    resultsfolder.joinpath("10foldcv_averaged.csv")
)
class2 = averaged10foldcv[averaged10foldcv["nClass"]==2]
class3 = averaged10foldcv[averaged10foldcv["nClass"]==3]
class4 = averaged10foldcv[averaged10foldcv["nClass"]==4]

sucrates = pd.read_csv(
    resultsfolder.joinpath("20_SDIsuccessrates.csv"), index_col=0
)
sucrates_class2 = sucrates.loc[2.0]
sucrates_class3 = sucrates.loc[3.0]
sucrates_class4 = sucrates.loc[4.0]

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

class2_df = fill_df(class2, sucrates_class2)
class3_df = fill_df(class3, sucrates_class3)
class4_df = fill_df(class4, sucrates_class4)
final_df = pd.concat([class2_df, class3_df, class4_df])

final_df.to_excel(resultsfolder.joinpath("SDI_latexTable.xlsx"))

sys.exit(0)

