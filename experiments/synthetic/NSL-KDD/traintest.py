import pickle
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 2:
    print("Possible usage: python3 traintest.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S"
]
feature_ranks = pd.read_csv(folder.joinpath("Combined/ranks.csv"), index_col=0)
feature_ranks = feature_ranks.rename(columns={"OA": "PDE-S"})

Xtrain = pd.read_csv(folder.joinpath("ProcessedCSV/Xtrain20.csv"), index_col=0).values
ytrain = pd.read_csv(folder.joinpath("ProcessedCSV/ytrain20.csv"), index_col=0)

Xtest  = pd.read_csv(folder.joinpath("ProcessedCSV/Xtest.csv"), index_col=0).values
ytest = pd.read_csv(folder.joinpath("ProcessedCSV/ytest.csv"), index_col=0)

results = pd.DataFrame(
    data=np.zeros((len(fsorder), 5)), index=fsorder,
    columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
)

for fs in fsorder:
    print(f" - Feature selection scheme: {fs}")
    top_features = (feature_ranks[fs].values).astype(np.int64)

    Xtrain_reduced = Xtrain[:, top_features]
    Xtest_reduced  = Xtest[:, top_features]

    # 3-fold grid search cross-validation
    CV_3fold = KFold(n_splits=3, shuffle=True, random_state=0)

    # kNN
    kNN = KNeighborsClassifier(weights="uniform")
    kNN_params = {'n_neighbors': [5,7,9]}
    clfkNN_GS = GridSearchCV(
        kNN, kNN_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
    )
    clfkNN_GS.fit(Xtrain_reduced, y_train)
    results.at[fs, "kNN"] = balanced_accuracy_score(
        ytest, clfkNN_GS.predict(Xtest_reduced)[0]
    )
    
    # SVM
    svm_clf = SVC()
    svm_params = {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
    clfsvm_GS = GridSearchCV(
        svm_clf, svm_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
    )
    clfsvm_GS.fit(Xtrain_reduced, y_train)
    results.at[fs, "SVM"] = balanced_accuracy_score(
        ytest, clfsvm_GS.predict(Xtest_reduced)[0]
    )
    
    # Gaussian Naive-Bayes
    naiveBayesClf = GaussianNB()
    naiveBayesClf.fit(Xtrain_reduced, y_train)
    results.at[fs, "Gaussian-NB"] = balanced_accuracy_score(
        ytest, naiveBayesClf.predict(Xtest_reduced)[0]
    )
    
    # LDA
    ldaClf = LinearDiscriminantAnalysis()
    ldaClf.fit(Xtrain_reduced, y_train)
    results.at[fs, "LDA"] = balanced_accuracy_score(
        ytest, ldaClf.predict(Xtest_reduced)[0]
    )

    # DT
    dt_clf = DecisionTreeClassifier(random_state=0)
    dt_params = {'splitter': ["best", "random"], 'max_depth': [3,5,7,9]}
    clfdt_GS = GridSearchCV(
        dt_clf, dt_params, cv=CV_3fold, scoring="balanced_accuracy"
    )
    clfdt_GS.fit(Xtrain_reduced, y_train)
    results.at[fs, "DT"] = balanced_accuracy_score(
        ytest, clfdt_GS.predict(Xtest_reduced)[0]
    )

sys.exit(0)
