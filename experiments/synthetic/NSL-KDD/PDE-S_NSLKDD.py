import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
    )
)
from pde_segregate import PDE_Segregate

if len(sys.argv) < 2:
    print("Possible usage: python3.11 PDE-S_NSLKDD.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

Xdf = pd.read_csv(folder.joinpath("Xtrain20.csv"), index_col=0)
X = Xdf.values

y = pd.read_csv(folder.joinpath("ytrain20.csv"), index_col=0)
y = np.reshape(y, -1)

Xdftest = pd.read_csv(folder.joinpath("Xtest.csv"), index_col=0)
Xtest = Xdftest.values

ytest = pd.read_csv(folder.joinpath("ytest.csv"), index_col=0)
ytest = np.reshape(ytest, -1)

X = np.delete(X, (19, 20), 1)
Xtest = np.delete(Xtest, (19, 20), 1)

# According to Canedo (2012), 40 % of 41 features = 16 features
nRetainedFeatures = 16

# === === === ===
# Carrying out feature selection for each dataset
elapsed_times = pd.Series(
    data=np.zeros(1),
    index=["PDE-S"]
)

scores_df = pd.DataFrame(
    data=np.zeros((X.shape[1], 2)),
    columns=[
        "feature",
        "PDE-S"
    ]
)
scores_df["feature"] = np.arange(0, X.shape[1], 1)

rank_df = pd.DataFrame(
    data=np.zeros((16, 2)),
    columns=[
        "rank",
        "PDE-S"
    ]
)
rank_df["rank"] = np.arange(0, 16, 1)

# Proposed algorithm
tPDE_start = process_time()
pdeSegregate = PDE_Segregate(
    integration_method="trapz", delta=500, bw_method="scott",
    n=2, n_jobs=-1, mode="release", lower_end=-1.5, upper_end=2.5
)
pdeSegregate.fit(X, y)
tPDE_stop = process_time()
tPDE = tPDE_stop - tPDE_start

# === === === === === === ===
# GET ELAPSED TIME
elapsed_times.at["PDE-S"] = tPDE

# === === === === === === ===
# GETTING TOP N FEATURES
rank_df.loc[:, "PDE-S"] = pdeSegregate.top_features_[
    :nRetainedFeatures
]

scores_df.loc[:, "PDE-S"] = pdeSegregate.feature_importances_

elapsed_times.to_csv("PDE-S_elapsed_times.csv")
rank_df.to_csv("PDE-S_rank.csv")
scores_df.to_csv("PDE-S_scores_df.csv")

sys.exit(0)
