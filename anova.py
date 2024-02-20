import numpy as np
from collections import defaultdict

class anova():
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.compute_variances()

    def compute_variances(self):
        # Grouping the samples according to unique y label
        self.y_segregatedGroup = self.segregateX_y()

        self.Fstat = np.zeros((self.X.shape[1],))
        self.EtaSq = np.zeros((self.X.shape[1],))

        # Grand sample size
        N = self.X.shape[0]

        # Degrees of freedom
        df_total = N-1
        df_between = len(self.y_segregatedGroup)-1
        df_within = N-len(self.y_segregatedGroup)

        for feat_idx in range(self.X.shape[1]):
            x = self.X[:,feat_idx]

            # Sum of squared deviations
            G = x.sum() # grand total

            T = [] # group totals
            n = [] # group sizes
            for y in self.y_segregatedGroup.keys():
                T.append(self.y_segregatedGroup[y][:,feat_idx].sum())
                n.append(len(self.y_segregatedGroup[y][:,feat_idx]))

            SS_total = (x**2).sum() - ((G**2)/N)

            SS_between = - (G**2)/N
            SS_within  = (x**2).sum()
            for i in range(len(T)):
                SS_between += (T[i]**2)/n[i]
                SS_within -= (T[i]**2)/n[i]
            
            # Mean squares of variability
            MS_between = SS_between / df_between
            MS_within  = SS_within / df_within
            
            # F-ratio
            F = MS_between / MS_within
            self.Fstat[feat_idx] = F
            
            # Proportion of Explained Variance (eta-squared)
            etaSquared = SS_between / SS_total
            self.EtaSq[feat_idx] = etaSquared

    def segregateX_y(self):
        """
        Routine to segregate X samples into unique y groups
        """
        unique_y = list(set(self.y))

        _subX = defaultdict()
        for uy in unique_y:
            _subX[uy] = self.X[np.where(self.y==uy)[0], :]

        return _subX
