import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 3:
    print("Possible usage: python3 loocv.py <processedDatasets> <selFeatures>")
    sys.exit(1)
else:
    processedDatasets_pkl = Path(sys.argv[1])
    selFeatures_pkl = Path(sys.argv[2])

with open(processedDatasets_pkl, "rb") as handle:
    processedDatasets_dict = pickle.load(handle)

with open(selFeatures_pkl, "rb") as handle:
    topFeatures = pickle.load(handle)
    
for ds in topFeatures.keys():
    print(f"Dataset: {ds}")
    ds_results = pd.DataFrame(columns=["kNN", "SVM", "Gaussian-NB", "LDA"])

    for fs in topFeatures[ds].keys():
        X = processedDatasets_dict[ds]['X']
        y = processedDatasets_dict[ds]['y']
        inds_topFeatures = topFeatures[ds][fs]

        X_reduced = X[:,inds_topFeatures]

        ypredArray_kNN = np.zeros((X.shape[0],))
        ypredArray_SVM = np.zeros((X.shape[0],))
        ypredArray_NBy = np.zeros((X.shape[0],))
        ypredArray_LDA = np.zeros((X.shape[0],))

        # LOOCV
        for n_val in range(X_reduced.shape[0]):
            X_validate = X_reduced[n_val,:]
            X_validate = X_validate.reshape((1, X_validate.shape[0]))

            train_inds = [n for n in range(X_reduced.shape[0]) if n != n_val]
            X_train = X_reduced[train_inds,:]
            y_train = y[train_inds]

            # 5-fold grid search cross-validation (default when not specified)
            # kNN
            kNN = KNeighborsClassifier(weights="uniform")
            kNN_params = {'n_neighbors': [5,7,9]}
            clfkNN_GS = GridSearchCV(kNN, kNN_params, scoring="balanced_accuracy")
            clfkNN_GS.fit(X_train, y_train)
            ypredArray_kNN[n_val] = clfkNN_GS.predict(X_validate)[0]

            # SVM
            svm_clf = SVC()
            svm_params = {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
            clfsvm_GS = GridSearchCV(svm_clf, svm_params, scoring="balanced_accuracy")
            clfsvm_GS.fit(X_train, y_train)
            ypredArray_SVM[n_val] = clfsvm_GS.predict(X_validate)[0]

            # Gaussian Naive-Bayes
            naiveBayesClf = GaussianNB()
            naiveBayesClf.fit(X_train, y_train)
            ypredArray_NBy[n_val] = naiveBayesClf.predict(X_validate)[0]

            # LDA
            ldaClf = LinearDiscriminantAnalysis()
            ldaClf.fit(X_train, y_train)
            ypredArray_LDA[n_val] = ldaClf.predict(X_validate)[0]

        balAcc_kNN = balanced_accuracy_score(y, ypredArray_kNN)
        balAcc_SVM = balanced_accuracy_score(y, ypredArray_SVM)
        balAcc_NBy = balanced_accuracy_score(y, ypredArray_NBy)
        balAcc_LDA = balanced_accuracy_score(y, ypredArray_LDA)
        ds_results = pd.concat(
            [
                ds_results,
                pd.DataFrame(
                    data=[[balAcc_kNN, balAcc_SVM, balAcc_NBy, balAcc_LDA]],
                    index=[fs], columns=["kNN", "SVM", "Gaussian-NB", "LDA"]
                )
            ]
        )

    # Saving results per dataset
    ds_results.to_csv(f"{ds}_LOOCV.csv")

sys.exit(0)
