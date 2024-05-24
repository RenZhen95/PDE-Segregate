import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif

from pdeseg import PDE_Segregate

# Dataset parameters
separation = [1, 3, 5]

fig, axs = plt.subplots(1, 3, figsize=(11.5, 5.3), sharey=True)

for i, s in enumerate(separation):
    print(f"Separation of class 1 from class 0: {s}")
    print("========================================")

    # Creating the Generator instance
    rng = np.random.default_rng(122807528840384100672342137672332424406)
    print(rng.random()) # as check

    n = 3000
    n0 = n
    n1 = n
    class0_sd = 1.0
    class1_sd = 1.0
    
    # Class 0
    class0 = rng.normal(loc=0.0, scale=class0_sd, size=n0)
    class0_calcsd = np.std(class0, ddof=1)
    
    # Class 1
    class1 = rng.normal(
        loc=s*class0_calcsd, scale=class1_sd, size=n1
    )
    
    # Initializing the dataset
    X = pd.DataFrame(
        data=np.zeros((n0+n1, 2)), columns=["Feature", "Class"]
    )
    X.iloc[:n0, 0] = class0
    
    X.iloc[n0:, 0] = class1
    X.iloc[n0:, 1] = 1
    
    y = X["Class"]
    X.drop(columns=["Class"], inplace=True)
    X = X.values
    pdeSegregate = PDE_Segregate(delta=1000)
    pdeSegregate.fit(X, y)
    
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
    
    SS_between = (T0**2)/n0 + (T1**2)/n1 - (G**2)/N
    SS_within  = (X**2).sum() - ((T0**2)/n0 + (T1**2)/n1)
    
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
        0, _ylim=(0.0, 5.25), _title=None,
        show_samples=False, _ax=axs[i], legend="intersection"
    )

    # Get means of normalized feature vectors
    mean0 = pdeSegregate.y_segregatedGroup[0.0][:,0].mean()
    mean1 = pdeSegregate.y_segregatedGroup[1.0][:,0].mean()
    meanTotal = np.append(
        pdeSegregate.y_segregatedGroup[0.0][:,0],
        pdeSegregate.y_segregatedGroup[1.0][:,0]
    )
    meanTotal = meanTotal.mean()

    axs[i].vlines(
        mean0, 0.0, 0.35, label="Mean of Class 0",
        color=np.array([76, 114, 176])/255, linewidth=2.5
    )
    axs[i].vlines(
        mean1, 0.0, 0.35, label="Mean of Class 1",
        color=np.array([221, 132, 82])/255, linewidth=2.5
    )
    axs[i].vlines(
        meanTotal, 0.0, 0.35, label="Total Mean", color="black",
        linewidth=2.5
    )
    xlabel = r"$\bar{X}$"
    xlabel += "\n"
    xlabel += r"Class Separation: $\sigma = $"
    xlabel += str(s)
    axs[i].set_xlabel(xlabel, fontsize='x-large')

    title = f"# Samples in Each Class: {n}\n"
    title += f"F-Ratio: {int(np.round(F))} | "
    title += r"$\eta^{2}$: "
    title += str(np.round(etaSquared, 3))
    axs[i].set_title(
        title, fontsize='x-large', horizontalalignment="left",
        loc="left"
    )

# Figures post-processing
for i in range(3):
    axs[i].grid(visible=True, which="major", axis="x")

axs[0].set_ylabel(r"$\hat{p}_X$     ", fontsize='large', rotation=0)

plt.tight_layout()
plt.show()

sys.exit()
