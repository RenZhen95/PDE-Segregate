import sys
import numpy as np
import pandas as pd
from pde_segregate import PDE_Segregate
from sklearn.feature_selection import f_classif

class0 = np.random.normal(loc=0.0, scale=1.0, size=1000)
class1 = np.random.normal(loc=2.0, scale=1.0, size=1000)

# Initializing the dataset
X = pd.DataFrame(data=np.zeros((2000, 2)), columns=["Feature", "Class"])
X.iloc[:1000, 0] = class0

X.iloc[1000:, 0] = class1
X.iloc[1000:, 1] = 1

y = X["Class"]
X.drop(columns=["Class"], inplace=True)
X = X.values
pdeSegregate = PDE_Segregate(X, y)

# Overlapping areas
print(f"OA from PDE-Segregate: {pdeSegregate.overlappingAreas[0]}")

# ANOVA F-Test
f_statistic, p_values = f_classif(X, y)
print(f"F-Stat from scikit-learn's ANOVA  : {f_statistic}")
print(f"P-Value from scikit-learn's ANOVA : {p_values}")

# # One-way ANOVA per hand
# # 1. Sum of squared deviations
# G = X.sum()       # grand total
# T0 = class0.sum() # group total of class 0
# T1 = class1.sum() # group total of class 1

# N = X.shape[0]       # grand sample size
# n0 = class0.shape[0] # sample size of class 0
# n1 = class1.shape[0] # sample size of class 1

# SS_total = (X**2).sum() - ((G**2)/N)

# SS_between = (T0**2)/n0 + (T1**2)/n1 - (G**2)/N
# SS_within  = (X**2).sum() - ((T0**2)/n0 + (T1**2)/n1)

# # 2. Degrees of freedom
# df_total = N-1
# df_between = 2-1
# df_within = N-2

# # 3. Mean squares of variability
# MS_between = SS_between / df_between
# MS_within  = SS_within / df_within

# # 4. F-ratio
# F = MS_between / MS_within
# print(f"\nManually computed F-Stat : {F}")

# # 5. Proportion of Explained Variance
# etaSquared = SS_between / SS_total
# print(f"Manually computed Eta^2  : {etaSquared}")

# Plot overlapping areas
pdeSegregate.plot_overlapAreas(
    0, feat_names=None, _ylim=None, _title=None, show_samples=True
)
