import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif

from pdeseg import PDE_Segregate

# ---------------------------------------------------------------- #
# MultiModal
rng = np.random.default_rng(122807528840384100672342137672332424406)
# Class 1
class1_cluster1 = rng.normal(loc=0.0, scale=0.1, size=2250)
class1_cluster2 = rng.normal(loc=0.5, scale=0.1, size=750)
class1mm = np.concatenate((class1_cluster1, class1_cluster2))

class1mm_calcsd = np.std(class1mm, ddof=1)

# Class 2
class2mm = rng.normal(loc=3*class1mm_calcsd, scale=0.14929775, size=3000)
# ---------------------------------------------------------------- #

# ---------------------------------------------------------------- #
# Normal
rng = np.random.default_rng(122807528840384100672342137672332424406)
# Class 1
class1 = rng.normal(loc=0.0, scale=1.0, size=3000)
class1_calcsd = np.std(class1, ddof=1)

# Class 2
class2 = rng.normal(loc=3*class1_calcsd, scale=1.0, size=3000)
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
    pdeSegregate = PDE_Segregate(k=2, lower_end=0.0, upper_end=1.0)
    pdeSegregate.fit(X, y)
    
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
        0, _combinations=None, _ax=_ax, legend="intersection",
    )

    # Get means of normalized feature vectors
    mean0 = pdeSegregate.y_segregatedGroup[0.0][0].mean()
    mean1 = pdeSegregate.y_segregatedGroup[1.0][0].mean()
    meanTotal = np.append(
        pdeSegregate.y_segregatedGroup[0.0][0], pdeSegregate.y_segregatedGroup[1.0][0]
    )
    meanTotal = meanTotal.mean()

    title = f"F-Ratio: {int(np.round(F))} | "
    title += r"$\eta^{2}$: "
    title += str(np.round(etaSquared, 3))
    _ax.set_title(
        title, horizontalalignment="left", loc="left"
    )

plotlength = 9.0
fig, axs = plt.subplots(
    2, 1, figsize=(plotlength/2, plotlength/1.33), sharex=True
)
_run(class1, class2, axs[0])
_run(class1mm, class2mm, axs[1])

# x-ticks
axs[0].set_xticks(np.arange(0.0, 1.2, 0.2))
axs[1].set_xticks(np.arange(0.0, 1.2, 0.2))

# x-label
axs[0].set_xlabel(r"$\boldsymbol{F'_{A}}$")
axs[1].set_xlabel(r"$\boldsymbol{F'_{B}}$")

# y-label
axs[0].set_ylabel(r"$\hat{p}$", fontsize='large', rotation=0, y=0.9, labelpad=9.0)
axs[1].set_ylabel(r"$\hat{p}$", fontsize='large', rotation=0, y=0.9, labelpad=9.0)

# turn on grid
axs[0].grid(visible=True, which="major", axis="both")
axs[1].grid(visible=True, which="major", axis="both")

plt.tight_layout()

plt.show()

sys.exit()
    
