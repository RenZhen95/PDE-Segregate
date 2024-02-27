import pickle
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 4:
    print(
        "Possible usage: python3 3foldcv_x20.py <processedDatasets> " +
        "<selFeatures> <saveFolder>"
    )
    sys.exit(1)
else:
    processedDatasets_pkl = Path(sys.argv[1])
    selFeatures_pkl = Path(sys.argv[2])
    saveFolder = Path(sys.argv[3])

with open(processedDatasets_pkl, "rb") as handle:
    processedDatasets_dict = pickle.load(handle)

with open(selFeatures_pkl, "rb") as handle:
    topFeatures = pickle.load(handle)

# Experiment with the following number of retained features
nRetainedFeatures = [25, 50, 75, 100]

for ds in topFeatures.keys():
    print(f"Dataset: {ds}")
    ds_results = pd.DataFrame(columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"])

    for fs in topFeatures[ds].keys():
        print(f" - Feature selection scheme: {fs}")
        if fs in ["RFEtry", "OAsilverman", "EtaSq"]:
            print("Skipping over RFEtry ...")
            continue

        for n_ in nRetainedFeatures:
            print(f" - Number of retained features: {n_}")
            X = processedDatasets_dict[ds]['X']
            y = processedDatasets_dict[ds]['y']
            inds_topFeatures = topFeatures[ds][fs][:n_]
            
            X_reduced = X[:,inds_topFeatures]
            
            balAcc_kNN = np.zeros((60,))
            balAcc_SVM = np.zeros((60,))
            balAcc_NB = np.zeros((60,))
            balAcc_LDA = np.zeros((60,))
            balAcc_DT  = np.zeros((60,))
            
            # 3-fold cv, 50 times
            rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=20)
            for i, (train_index, test_index) in enumerate(rskf.split(X_reduced, y)):
                X_validate = X_reduced[test_index,:]
                y_validate = y[test_index]

                X_train = X_reduced[train_index,:]
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
                balAcc_kNN[i] = balanced_accuracy_score(
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
                balAcc_SVM[i] = balanced_accuracy_score(
                    y_validate, clfsvm_GS.predict(X_validate)
                )
            
                # Gaussian Naive-Bayes
                naiveBayesClf = GaussianNB()
                naiveBayesClf.fit(X_train, y_train)
                balAcc_NB[i] = balanced_accuracy_score(
                    y_validate, naiveBayesClf.predict(X_validate)
                )
            
                # LDA
                ldaClf = LinearDiscriminantAnalysis()
                ldaClf.fit(X_train, y_train)
                balAcc_LDA[i] = balanced_accuracy_score(
                    y_validate, ldaClf.predict(X_validate)
                )

                # DT
                dt_clf = DecisionTreeClassifier(random_state=0)
                dt_params = {'splitter': ["best","random"], 'max_depth': [3,5,7,9]}
                clfdt_GS = GridSearchCV(
                    dt_clf, dt_params, cv=CV_3fold, scoring="balanced_accuracy"
                )
                clfdt_GS.fit(X_train, y_train)
                balAcc_DT[i] = balanced_accuracy_score(
                    y_validate, clfdt_GS.predict(X_validate)
                )
                
            balAccMean_kNN = balAcc_kNN.mean()
            balAccMean_SVM = balAcc_SVM.mean()
            balAccMean_NB  = balAcc_NB.mean()
            balAccMean_LDA = balAcc_LDA.mean()
            balAccMean_DT  = balAcc_DT.mean()

            ds_results = pd.concat(
                [
                    ds_results,
                    pd.DataFrame(
                        data=[[balAccMean_kNN, balAccMean_SVM, balAccMean_NB, balAccMean_LDA, balAccMean_DT]],
                        index=[f"{fs}-{n_}"], columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
                    )
                ]
            )

    # Saving results per dataset
    ds_results.to_csv(saveFolder.joinpath(f"{ds}_3foldCVx20.csv"))

sys.exit(0)
