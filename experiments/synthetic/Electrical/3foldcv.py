import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 3:
    print(
        "Possible usage: python3.11 3foldcv.py <ElectricalFolder> <datasetName>"
    )
    sys.exit(1)
else:
    ElectricalFolder = Path(sys.argv[1])
    datasetName = sys.argv[2]

# Loading datasets
with open(ElectricalFolder.joinpath(f"{datasetName}_datasets.pkl"), "rb") as handle:
    datasets = pickle.load(handle)

# Reading the top 10 features
resultsFolder = ElectricalFolder.joinpath("Combined")
ranks_df = pd.read_csv(
    resultsFolder.joinpath(f"{datasetName}_ranks.csv"), index_col=0
)

ranks_n30 = ranks_df[ranks_df["n_obs"] == 30.0]
ranks_n50 = ranks_df[ranks_df["n_obs"] == 50.0]
ranks_n70 = ranks_df[ranks_df["n_obs"] == 70.0]
ranks = {30: ranks_n30, 50: ranks_n50, 70: ranks_n70}

fs_methods = ["RlfF", "MSurf", "RFGini", "MI", "FT", "OA", "OApw", "IRlf", "LHRlf"]

averaged_performance_df = pd.DataFrame(
    data=np.zeros((2, 9)), index=["Mean", "S.D"], columns=fs_methods
)

# 3 nObs x 50 iterations x 9 FS x 5 Classifiers
performance_df = pd.DataFrame(
    data=np.zeros((3*50*9*5, 5)), columns=["Bal.Acc", "nObs", "Iteration", "FS", "Clf"]
)

skf = StratifiedKFold(n_splits=3)

count = 0
for nObs in [30, 50, 70]:
    for itr in range(50):
        ranks_peritr = ranks[nObs][ranks[nObs]["iteration"] == itr]
        ranks_peritr = ranks_peritr.drop(
            columns=["rank", "iteration", "n_obs"]
        )

        X = datasets[nObs][itr]['X']
        y = datasets[nObs][itr]['y']

        for fs in fs_methods:
            top_features = ranks_peritr[fs].to_numpy()
            top_features = list(map(int, top_features))
            X_reduced = X[:,top_features]

            balAcc_kNN = np.zeros(3)
            balAcc_SVM = np.zeros(3)
            balAcc_NB  = np.zeros(3)
            balAcc_LDA = np.zeros(3)
            balAcc_DT  = np.zeros(3)

            # Carry out stratified k-fold
            for fold, (train_index, test_index) in enumerate(skf.split(X_reduced, y)):
                X_validate = X_reduced[test_index, :]
                y_validate = y[test_index]

                X_train = X_reduced[train_index, :]
                y_train = y[train_index]

                # 3-fold grid search cross-validation
                CV_3fold = KFold(n_splits=3, shuffle=True, random_state=0)

                # kNN
                kNN = KNeighborsClassifier(weights="uniform")
                kNN_params = {'n_neighbors': [5,7,9]}
                clfkNN_GS = GridSearchCV(
                    kNN, kNN_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
                )
                clfkNN_GS.fit(X_train, y_train)
                balAcc_kNN[fold] = balanced_accuracy_score(
                    y_validate, clfkNN_GS.predict(X_validate)
                )

                # SVM
                svm_clf = SVC()
                svm_params = {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
                clfsvm_GS = GridSearchCV(
                    svm_clf, svm_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy",
                    error_score="raise"
                )
                clfsvm_GS.fit(X_train, y_train)
                balAcc_SVM[fold] = balanced_accuracy_score(
                    y_validate, clfsvm_GS.predict(X_validate)
                )

                # Gaussian Naive-Bayes
                naiveBayesClf = GaussianNB()
                naiveBayesClf.fit(X_train, y_train)
                balAcc_NB[fold] = balanced_accuracy_score(
                    y_validate, naiveBayesClf.predict(X_validate)
                )

                # LDA
                ldaClf = LinearDiscriminantAnalysis()
                ldaClf.fit(X_train, y_train)
                balAcc_LDA[fold] = balanced_accuracy_score(
                    y_validate, ldaClf.predict(X_validate)
                )

                # DT
                dt_clf = DecisionTreeClassifier(random_state=0)
                dt_params = {'splitter': ["best","random"], 'max_depth': [3,5,7,9]}
                clfdt_GS = GridSearchCV(
                    dt_clf, dt_params, cv=CV_3fold, scoring="balanced_accuracy"
                )
                clfdt_GS.fit(X_train, y_train)
                balAcc_DT[fold] = balanced_accuracy_score(
                    y_validate, clfdt_GS.predict(X_validate)
                )

            performance_df.at[count, "Bal.Acc"] = balAcc_kNN.mean()
            performance_df.at[count, "nObs"] = nObs
            performance_df.at[count, "Iteration"] = itr
            performance_df.at[count, "FS"] = fs
            performance_df.at[count, "Clf"] = "kNN"

            performance_df.at[count+1, "Bal.Acc"] = balAcc_SVM.mean()
            performance_df.at[count+1, "nObs"] = nObs
            performance_df.at[count+1, "Iteration"] = itr
            performance_df.at[count+1, "FS"] = fs
            performance_df.at[count+1, "Clf"] = "SVM"

            performance_df.at[count+2, "Bal.Acc"] = balAcc_NB.mean()
            performance_df.at[count+2, "nObs"] = nObs
            performance_df.at[count+2, "Iteration"] = itr
            performance_df.at[count+2, "FS"] = fs
            performance_df.at[count+2, "Clf"] = "NB"

            performance_df.at[count+3, "Bal.Acc"] = balAcc_LDA.mean()
            performance_df.at[count+3, "nObs"] = nObs
            performance_df.at[count+3, "Iteration"] = itr
            performance_df.at[count+3, "FS"] = fs
            performance_df.at[count+3, "Clf"] = "LDA"

            performance_df.at[count+4, "Bal.Acc"] = balAcc_DT.mean()
            performance_df.at[count+4, "nObs"] = nObs
            performance_df.at[count+4, "Iteration"] = itr
            performance_df.at[count+4, "FS"] = fs
            performance_df.at[count+4, "Clf"] = "DT"
            count += 5

sys.exit(0)
