import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif

sys.path.append(os.path.dirname(os.getcwd()))
from pde_segregate import PDE_Segregate

# ---------------------------------------------------------------- #
# MultiModal
rng = np.random.default_rng(122807528840384100672342137672332424406)
# Class 0
class0_cluster1 = rng.normal(loc=0.0, scale=0.1, size=2250)
class0_cluster2 = rng.normal(loc=0.5, scale=0.1, size=750)
class0mm = np.concatenate((class0_cluster1, class0_cluster2))

class0mm_calcsd = np.std(class0mm, ddof=1)

# Class 1
class1mm = rng.normal(loc=3*class0mm_calcsd, scale=0.14929775, size=3000)
# ---------------------------------------------------------------- #

# ---------------------------------------------------------------- #
# Normal
rng = np.random.default_rng(122807528840384100672342137672332424406)
# Class 0
class0 = rng.normal(loc=0.0, scale=1.0, size=3000)
class0_calcsd = np.std(class0, ddof=1)

# Class 1
class1 = rng.normal(loc=3*class0_calcsd, scale=1.0, size=3000)
# ---------------------------------------------------------------- #

def _run(c0, c1, _ax):
    # Initializing the dataset
    X = pd.DataFrame(
        data=np.zeros((len(c0)+len(c1), 2)), columns=["Feature", "Class"]
    )
    X.iloc[:len(c0), 0] = c0
    
    X.iloc[len(c0):, 0] = c1
    X.iloc[len(c0):, 1] = 1
    
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
    T0 = c0.sum() # group total of class 0
    T1 = c1.sum() # group total of class 1
    
    N = X.shape[0]       # grand sample size
    
    SS_total = (X**2).sum() - ((G**2)/N)
    
    SS_between = (T0**2)/len(c0) + (T1**2)/len(c1) - (G**2)/N
    SS_within  = (X**2).sum() - ((T0**2)/len(c0) + (T1**2)/len(c1))
    
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
        0, feat_names='X', _ylim=(0.0, 5.0),
        show_samples=True, _ax=_ax
    )

    return pdeSegregate.overlappingAreas[0]

fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.3))
oa = _run(class0, class1, axs[0])
oamm = _run(class0mm, class1mm, axs[1])

axs[0].set_title(f"Intersection area: {np.round(oa, 3)}")
axs[1].set_title(f"Intersection area: {np.round(oamm, 3)}")

axs[0].grid(visible=True)
axs[1].grid(visible=True)

plt.tight_layout()
plt.show()

sys.exit()
    
