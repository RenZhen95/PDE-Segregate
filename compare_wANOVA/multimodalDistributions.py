import os, sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

sys.path.append(os.path.dirname(os.getcwd()))
from pde_segregate import PDE_Segregate

# Creating the Generator instance
rng = np.random.default_rng(122807528840384100672342137672332424406)
print(rng.random()) # as check

# Class 0
class0_cluster1 = rng.normal(loc=0.0, scale=0.1, size=2250)
class0_cluster2 = rng.normal(loc=0.5, scale=0.1, size=750)
class0 = np.concatenate((class0_cluster1, class0_cluster2))

class0_calcsd = np.std(class0, ddof=1)

# Class 1
class1 = rng.normal(loc=3*class0_calcsd, scale=0.14929775, size=3000)

# Initializing the dataset
X = pd.DataFrame(
    data=np.zeros((len(class0)+len(class1), 2)), columns=["Feature", "Class"]
)
X.iloc[:len(class0), 0] = class0

X.iloc[len(class0):, 0] = class1
X.iloc[len(class0):, 1] = 1

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

# One-way ANOVA per hand
# 1. Sum of squared deviations
G = X.sum()       # grand total
T0 = class0.sum() # group total of class 0
T1 = class1.sum() # group total of class 1

N = X.shape[0]       # grand sample size

SS_total = (X**2).sum() - ((G**2)/N)

SS_between = (T0**2)/len(class0) + (T1**2)/len(class1) - (G**2)/N
SS_within  = (X**2).sum() - ((T0**2)/len(class0) + (T1**2)/len(class1))

# 2. Degrees of freedom
df_total = N-1
df_between = 2-1
df_within = N-2

# 3. Mean squares of variability
MS_between = SS_between / df_between
MS_within  = SS_within / df_within

# 4. F-ratio
F = MS_between / MS_within
print(f"\nManually computed F-Stat : {F}")

# 5. Proportion of Explained Variance
etaSquared = SS_between / SS_total
print(f"Manually computed Eta^2  : {etaSquared}")

# Plot overlapping areas
pdeSegregate.plot_overlapAreas(
    0, feat_names='X', _ylim=None, _title=None, show_samples=True,
    savefig_title=None
)

sys.exit()
    
