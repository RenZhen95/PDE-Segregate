import sys
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from collections import defaultdict

with open(sys.argv[1], "rb") as handle:
    datasets_dict = pickle.load(handle)

for k in datasets_dict.keys():
    print(f"---\n{k}")
    X = datasets_dict[k]['X']
    Y = datasets_dict[k]['y']

    n_uniquey = defaultdict(int)
    for y in Y:
        n_uniquey[y] += 1

    print(X.shape)
    n_uniquey = dict(sorted(n_uniquey.items()))
    percentage_uniquey = defaultdict(float)
    for y in n_uniquey:
        percentage_uniquey[y] = np.round((n_uniquey[y] / X.shape[0])*100)
        print(f"{y}: {percentage_uniquey[y]}")
        
    print("---\n")

# # 
# df = pd.read_csv(sys.argv[2], index_col=0)
# print(df)

df = loadmat(sys.argv[2])
print(df.keys())
# print(df["data"][:,0])
print(df["X"].shape)

# # Check features with zero standard deviation
# X = zscore(X, axis=0, ddof=1, nan_policy='propagate')

# # Remove features with nan
# nanCheck = np.isnan(X)
# if np.any(nanCheck):
#     nan_y = np.where(nanCheck)
#     nan_y = list(set(nan_y[1]))

# print(nan_y)
#     # X_st = np.delete(X_st, nan_y, 1)
            
# for feat_idx in range(df["X"].shape[1]):
#     if df["X"][:, feat_idx].std(ddof=1) == 0.0:
#         print(feat_idx)
