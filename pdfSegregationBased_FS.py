import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde

def segregateX_y(X, y):
    unique_y = list(set(y))
    
    _subX = defaultdict()
    for uy in unique_y:
        _subX[uy] = X[np.where(y==uy)[0], :]

    return _subX

def plot_overlapAreas(_kernels, _lengths, computedOA, _featName, _title, _ax, _ylim=None):
    yStack = []
    _xGrid = np.linspace(0, 1, max(_lengths))

    for k in _kernels:
        Y = np.reshape(k[1](_xGrid).T, _xGrid.shape)
        yStack.append(Y)

        _ax.plot(_xGrid, Y, label=k[0])

    yIntersection = np.amin(yStack, axis=0)

    fill_poly = _ax.fill_between(
        _xGrid, 0, yIntersection, label=f'Intersection: {round(computedOA,3)}'
    )
    fill_poly.set_hatch('xxx')

    _ax.set_xlabel(_featName)
    _ax.set_xlim((0.0, 1.0))
    _ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    if not _ylim is None:
        _ax.set_ylim(_ylim)

    _ax.legend(bbox_to_anchor=(0.8,1), loc='upper left', title=_title, fontsize='small')

def compute_OA(yLabels, y_segregatedGroup, feat_idx):
    lengths = []; kernels = []
    
    for idx, y in enumerate(yLabels):
        if idx == 0:
            total_y = y_segregatedGroup[y][:, feat_idx]
        else:
            total_y = np.append(total_y, y_segregatedGroup[y][:, feat_idx])

    # Parameters to "center" the series
    minVal = total_y.min()
    maxValCentered = (total_y - minVal).max()
    
    for y in yLabels:
        X_feat_idx = y_segregatedGroup[y][:, feat_idx]

        # Centering the series
        # 1. Subtract series with series.min()
        # 2. Divide series by its max
        X_feat_idxNormalized = X_feat_idx - minVal
        X_feat_idxNormalized = X_feat_idxNormalized / maxValCentered
    
        # If X has a standard deviation of zero, then we are dealing with a "Dirac
        # function", which technically has an area bounded by the x-axis of zero and
        # hence will be ignored
        if X_feat_idxNormalized.std() != 0.0:
            lengths.append(X_feat_idxNormalized.shape[0])
    
            kernel = gaussian_kde(X_feat_idxNormalized)
            kernels.append((y, kernel))
        else:
            print(
                f"Feature (idx:{feat_idx}) for samples with y={y} " +
                "exhibits zero standard deviation!"
            )
    
    # There must be AT LEAST two target groups with non-zero S.D, or this feature
    # ranking method just makes no sense
    if len(lengths) < 2:
        print(
            f"\nThe feature (idx:{feat_idx}) has LESS than two feature groups with " +
            "non-zero S.D and thus not suitable for this feature selection method. " +
            "Overlapping area of PDF will simply be assigned a value of 10.0"
        )
        OA = 10.0
        time.sleep(10)
    else:
        # Initializing the x-axis grid
        XGrid = np.linspace(0, 1, max(lengths))
    
        yStack = []
        for k in kernels:
            Y = np.reshape(k[1](XGrid).T, XGrid.shape)
            yStack.append(Y)
    
        yIntersection = np.amin(yStack, axis=0)
        OA = np.trapz(yIntersection, XGrid)

    return OA, kernels, lengths

def get_overlappingAreasofPDF(X, y):
    # Grouping the samples according to unique y label
    y_segregatedGroup = segregateX_y(X, y)
    
    # You can't build a distribution function with only one sample
    yToRemove = []
    
    for y in y_segregatedGroup.keys():
        if y_segregatedGroup[y].shape[0] < 2:
            yToRemove.append(score)
            print(f"---\ny={y} sub-dataset has only 1 sample and will be excluded ... ")
    
    if len(yToRemove) != 0:
        for y in yToRemove:
            y_segregatedGroup.pop(y, None)
            
    if len(y_segregatedGroup) == 1:
        raise ValueError(f"There's only one target label, y={y_segregatedGroup.keys()}")
    
    # Intra-feature normalization and computing the overlap area among classes
    yLabels = list(y_segregatedGroup.keys())
    yLabels.sort()
    
    overlapScores = np.zeros((X.shape[1],))
    for feat_idx in range(X.shape[1]):
        OA, kernels, lengths = compute_OA(yLabels, y_segregatedGroup, feat_idx)
        overlapScores[feat_idx] = OA

    # for feat_idx in range(X.shape[1]):
    #     lengths = []; kernels = []
    
    #     for idx, y in enumerate(yLabels):
    #         if idx == 0:
    #             total_y = y_segregatedGroup[y][:, feat_idx]
    #         else:
    #             total_y = np.append(total_y, y_segregatedGroup[y][:, feat_idx])

    #     # Parameters to "center" the series
    #     minVal = total_y.min()
    #     maxValCentered = (total_y - minVal).max()
    
    #     for y in yLabels:
    #         X_feat_idx = y_segregatedGroup[y][:, feat_idx]

    #         # Centering the series
    #         # 1. Subtract series with series.min()
    #         # 2. Divide series by its max
    #         X_feat_idxNormalized = X_feat_idx - minVal
    #         X_feat_idxNormalized = X_feat_idxNormalized / maxValCentered
    
    #         # If X has a standard deviation of zero, then we are dealing with a "Dirac
    #         # function", which technically has an area bounded by the x-axis of zero and
    #         # hence will be ignored
    #         if X_feat_idxNormalized.std() != 0.0:
    #             lengths.append(X_feat_idxNormalized.shape[0])
    
    #             kernel = gaussian_kde(X_feat_idxNormalized)
    #             kernels.append((y, kernel))
    #         else:
    #             print(
    #                 f"Feature (idx:{feat_idx}) for samples with y={y} " +
    #                 "exhibits zero standard deviation!"
    #             )
        
    #     # There must be AT LEAST two target groups with non-zero S.D, or this feature
    #     # ranking method just makes no sense
    #     if len(lengths) < 2:
    #         print(
    #             f"\nThe feature (idx:{feat_idx}) has LESS than two feature groups with " +
    #             "non-zero S.D and thus not suitable for this feature selection method. " +
    #             "Overlapping area of PDF will simply be assigned a value of 10.0"
    #         )
    #         overlapScores[feat_idx] = 10.0
    #         time.sleep(10)
    #     else:
    #         # Initializing the x-axis grid
    #         XGrid = np.linspace(0, 1, max(lengths))
    
    #         yStack = []
    #         for k in kernels:
    #             Y = np.reshape(k[1](XGrid).T, XGrid.shape)
    #             yStack.append(Y)
    
    #         yIntersection = np.amin(yStack, axis=0)
    #         overlapArea = np.trapz(yIntersection, XGrid)
    #         overlapScores[feat_idx] = overlapArea

    return overlapScores
    
