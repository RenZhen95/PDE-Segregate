import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time

from skrebate import ReliefF, MultiSURF
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
    )
)
from pde_segregate import PDE_Segregate

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 2:
    print("Possible usage: python3.11 experiment.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

Xdf = pd.read_csv(folder.joinpath("Xtrain.csv"), index_col=0)
X = Xdf.values
y = pd.read_csv(folder.joinpath("ytrain.csv"), index_col=0)
y = np.reshape(y, -1)

Xdftest = pd.read_csv(folder.joinpath("Xtest.csv"), index_col=0)
Xtest = Xdftest.values
ytest = pd.read_csv(folder.joinpath("ytest.csv"), index_col=0)
ytest = np.reshape(ytest, -1)

# According to Canedo (2012), 40 % of 41 features = 16 features

# === === === === === === ===
# FEATURE RANKING METHODS
# From scikit-rebate (https://github.com/EpistasisLab/scikit-rebate)
# ReliefF
tRlfF_start = process_time()
RlfF = ReliefF(n_neighbors=7, n_jobs=-1) # From Cai, 2014
RlfF.fit(X, y)
tRlfF_stop = process_time()
tRlfF = tRlfF_stop - tRlfF_start
print(RlfF.feature_importances_)
sys.exit()
# MultiSURF
tMSurf_start = process_time()
MSurf = MultiSURF(n_jobs=-1)
MSurf.fit(X, y)
tMSurf_stop = process_time()
tMSurf = tMSurf_stop - tMSurf_start

# From scikit-learn
# Mutual Information
tMI_start = process_time()
resMI = mutual_info_classif(X, y, n_neighbors=7, random_state=0)
tMI_stop = process_time()
tMI = tMI_stop - tMI_start

# ANOVA F-value
tFT_start = process_time()
resFT_stat, resFT_p = f_classif(X, y)
tFT_stop = process_time()
tFT = tFT_stop - tFT_start

# Random forest ensemble data mining to increase information gain/reduce impurity
tRF_start = process_time()
rfGini = RandomForestClassifier(
    n_estimators=1000, criterion="gini", random_state=0, n_jobs=-1
)
rfGini.fit(X, y)
tRF_stop = process_time()
tRF = tRF_stop - tRF_start

# Proposed algorithm
# Overlapping Areas of PDEs (total)
tPDE_start = process_time()
pdeSegregate = PDE_Segregate(
    integration_method="trapz", delta=1500, bw_method="scott",
    pairwise=False, n_jobs=-1
)
pdeSegregate.fit(X, y)
tPDE_stop = process_time()
tPDE = tPDE_stop - tPDE_start

# Overlapping Areas of PDEs (pairwise)
tPDEpw_start = process_time()
pdeSegregatePW = PDE_Segregate(
    integration_method="trapz", delta=1500, bw_method="scott",
    pairwise=True, n_jobs=-1
)
pdeSegregatePW.fit(X, y)
tPDEpw_stop = process_time()
tPDEpw = tPDEpw_stop - tPDEpw_start

# === === === === === === ===
# GETTING TOP N FEATURES
rank_df.loc[count_r:count_r+119, "RlfF"] = get_indsTopnFeatures(
    RlfF.feature_importances_, nRetainedFeatures
)
rank_df.loc[count_r:count_r+119, "MSurf"] = get_indsTopnFeatures(
    MSurf.feature_importances_, nRetainedFeatures
)
rank_df.loc[count_r:count_r+119, "MI"] = get_indsTopnFeatures(
    resMI, nRetainedFeatures
)
rank_df.loc[count_r:count_r+119, "RFGini"] = get_indsTopnFeatures(
    rfGini.feature_importances_, nRetainedFeatures
)
rank_df.loc[count_r:count_r+119, "FT"] = get_indsTopnFeatures(
    resFT_stat, nRetainedFeatures
)
rank_df.loc[count_r:count_r+119, "OA"] = pdeSegregate.top_features_[
    :nRetainedFeatures
]
rank_df.loc[count_r:count_r+119, "OApw"] = pdeSegregatePW.top_features_[
    :nRetainedFeatures
]
rank_df.loc[count_r:count_r+119, "iteration"] = np.repeat([d_itr], 120)
rank_df.loc[count_r:count_r+119, "nClass"] = np.repeat([nClass], 120)

sys.exit(0)
