import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde

class PDE_Segregate():
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.compute_PDEoverlappingAreas()

    def get_scores(self):
        """
        Get feature importance based on feature's ability to segregate the
        PDEs of the class-segregated samples. Returns the negative overlapping
        areas of the PDEs, such that the lower the score (i.e. the more
        negative the area), the less important the feature
        """
        return -1*self.overlappingAreas


    def compute_PDEoverlappingAreas(self):
        """
        Get the overlapping areas of the PDE of class-segregated groups
        """
        # Grouping the samples according to unique y label
        self.y_segregatedGroup = self.segregateX_y()

        # You can't build a distribution function with only one sample
        yToRemove = []

        for y in self.y_segregatedGroup.keys():
            if self.y_segregatedGroup[y].shape[0] < 2:
                yToRemove.append(score)
                print(
                    f"---\ny={y} sub-dataset has only 1 sample and will be " +
                    "excluded ... "
                )

        # Removing features that have class groups with only two samples
        if len(yToRemove) != 0:
            for y in yToRemove:
                self.y_segregatedGroup.pop(y, None)

        # Removing features that only exhibit one target group after
        # pre-processing
        if len(self.y_segregatedGroup) == 1:
            raise ValueError(
                "There's only one target label, " +
                f"y={self.y_segregatedGroup.keys()}"
            )

        # Computing the overlap area among classes
        self.yLabels = list(self.y_segregatedGroup.keys())
        self.yLabels.sort()

        self.overlappingAreas = np.zeros((self.X.shape[1],))
        for feat_idx in range(self.X.shape[1]):
            OA, kernels, lengths = self.compute_OA(feat_idx)
            self.overlappingAreas[feat_idx] = OA

    def get_topnFeatures(self, n):
        """
        Returns the indices of the top n features
        """
        res_tmp = self.overlappingAreas
        res_tmpSorted = np.sort(res_tmp)

        inds_topFeatures = []

        counter = 1

        OAs = []

        for i in res_tmpSorted:
            OAs.append(i)
            if i in OAs:
                pass

            sel_inds = np.where(res_tmp==i)[0]

            # Dealing with overlapping areas, which multiple features
            # share (e.g. 0.0)
            if len(sel_inds) == 1:
                inds_topFeatures.append(sel_inds[0])
                counter += 1
            else:
                print(
                    f"The following {len(sel_inds)} features:\n{sel_inds}" +
                    "\nhave the same computed overlapping intersection areas: " +
                    f"{i}"
                )
                for ind in sel_inds:
                    inds_topFeatures.append(ind)
                    counter += 1
                    if counter==n+1:
                        break

            if counter==n+1:
                break

        return inds_topFeatures

    def compute_OA(self, feat_idx):
        """
        Compute the overlapping areas of the PDE of class-segregated groups
        for a given feature
        """
        lengths = []
        kernels = []

        for idx, y in enumerate(self.yLabels):
            if idx == 0:
                total_y = self.y_segregatedGroup[y][:, feat_idx]
            else:
                total_y = np.append(
                    total_y, self.y_segregatedGroup[y][:, feat_idx]
                )

        # Parameters to "center" the series
        minVal = total_y.min()
        maxValCentered = (total_y - minVal).max()

        for y in self.yLabels:
            X_feat_idx = self.y_segregatedGroup[y][:, feat_idx]

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
                "Overlapping area of PDE will simply be assigned a value of 10.0"
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

    def segregateX_y(self):
        """
        Routine to segregate X samples into unique y groups
        """
        unique_y = list(set(self.y))

        _subX = defaultdict()
        for uy in unique_y:
            _subX[uy] = self.X[np.where(self.y==uy)[0], :]

        return _subX

    def plot_overlapAreas(
            self, feat_idx, feat_names=None, _ylim=None, _title=None
    ):
        """
        Function to plot overlapping areas for a given feature
        """
        OA, _kernels, _lengths = self.compute_OA(feat_idx)

        yStack = []
        _xGrid = np.linspace(0, 1, max(_lengths))

        fig, _ax = plt.subplots(1,1)
        for k in _kernels:
            Y = np.reshape(k[1](_xGrid).T, _xGrid.shape)
            yStack.append(Y)

            _ax.plot(_xGrid, Y, label=k[0])

        yIntersection = np.amin(yStack, axis=0)

        fill_poly = _ax.fill_between(
            _xGrid, 0, yIntersection, label=f'Intersection: {round(OA, 3)}'
        )
        fill_poly.set_hatch('xxx')

        if not feat_names is None:
            _ax.set_xlabel(feat_names[feat_idx])
        else:
            _ax.set_xlabel(f"Feature {feat_idx}")

        _ax.set_xlim((0.0, 1.0))
        _ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        if not _ylim is None:
            _ax.set_ylim(_ylim)

        _ax.legend(bbox_to_anchor=(0.8,1), loc='upper left', title=_title, fontsize='small')

        plt.show()
