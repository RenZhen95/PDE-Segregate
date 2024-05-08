import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

class PDE_Segregate():
    def __init__(
            self, integration_method="trapz", delta=1500,
            bw_method="scott", n=2, n_jobs=1, mode="release"
    ):
        """
        Parameters
        ----------
        integration_method : str
         - Integration method.

           Available options include 'numpy.trapz' (default) and 'sum'.

        delta : int
         - Number of cells in the x-grid

        bw_method : str, scalar or callable
         - The method used to calculate the estimator bandwith. This can be
           'scott' and 'silverman', a scalar constant or a callable. For
           more details, see scipy.stats.gaussian_kde documentation.

        n : intpairwise : bool
         - Compute the mean intersection area between N choose 2 combinations
           of intersection areas.

        n_jobs : int
         - Number of processors to use. -1 to use all available processors.

        mode : str ("release", "development")
         - Option implemented during development to return constructed kernels
           PDEs.
        """
        self.integration_method = integration_method
        self.delta = delta
        self.bw_method = bw_method
        self.n = n
        self.n_jobs = n_jobs
        self.mode = mode

        # Initializing the x-axis grid
        # self.XGrid = np.linspace(0.0, 1.0, self.delta)
        self.XGrid = np.linspace(-1.0, 2.0, self.delta)

    def fit(self, X, y):
        """
        Get the intersection areas of the PDE of class-segregated groups.

        Parameters
        ----------
        X : np.array
         - Dataset with the shape: (n_samples, n_features)

        y : np.array
         - Class vector
        """
        self.X = X
        self.y = y

        # Grouping the samples according to unique y label
        self.y_segregatedGroup = self.segregateX_y()

        # Do not allow user to use PDE-Segregate, when class has only one sample
        yToRemove = []
        for y in self.y_segregatedGroup.keys():
            if self.y_segregatedGroup[y].shape[0] == 1:
                yToRemove.append(y)
                print(
                    f"---\ny={y} sub-dataset has only 1 sample and will be " +
                    "excluded ... "
                )
        #  - removing class populations with only one sample
        if len(yToRemove) != 0:
            for y in yToRemove:
                self.y_segregatedGroup.pop(y, None)
        #  - abort if all remaining samples belong to only one single class
        if len(self.y_segregatedGroup) == 1:
            raise ValueError(
                "There's only one target label, " +
                f"y={self.y_segregatedGroup.keys()}"
            )

        # Initializing a list of available classes
        self.yLabels = list(self.y_segregatedGroup.keys())
        self.yLabels.sort()

        # Construct kernel density estimator per class for every feature
        delayed_calls = (
            delayed(
                self.construct_kernel
            )(feat_idx) for feat_idx in range(self.X.shape[1])
        )
        res = Parallel(n_jobs=self.n_jobs)(delayed_calls)
        if self.mode == "development":
            self.feature_kernels = [item[0] for item in res]
            self.pdes = [item[1] for item in res]
        elif self.mode == "release":
            self.pdes = res

        print("Kernels constructed ... ")

        # Compute intersection areas

        # Number of different combinations: nclass choose n
        _combinations = combinations(self.yLabels, self.n)
        cStack = []
        for c in _combinations:
            delayed_calls_intersectionArea = (
                delayed(
                    self.compute_intersectionArea
                )(feat_idx, c) for feat_idx in range(self.X.shape[1])
            )
            c_intersection = Parallel(
                n_jobs=self.n_jobs, backend="threading", verbose=1
            )(delayed_calls_intersectionArea)

            cStack.append(c_intersection)

        cStack = np.array(cStack)
        self.intersectionAreas = np.mean(cStack, axis=0)

        # Get feature importances as expressed in terms of reciprocal of
        # computed intersection areas
        self.feature_importances_ = 1/self.intersectionAreas

        # Get rankings of features ordered from most important to least
        self.top_features_ = self.get_topnFeatures(self.X.shape[1])

    def compute_intersectionArea(self, feat_idx, pairwise):
        """
        Compute intersection area between estimated PDEs.

        Parameters
        ----------
        feat_idx : int
         - Index of the desired feature in the given dataset, X.

        pairwise : tuple
         - Pair of indices indicating which pair of classes to compare.

        Returns
        -------
        OA : float
         - Computed intersection area of the PDEs.
        """
        yStack = []

        # if not pairwise:
        #     for Y in self.feature_kernels[feat_idx].values():
        #         yStack.append(Y)

        Y0 = self.pdes[feat_idx][pairwise[0]]
        yStack.append(Y0)

        Y1 = self.pdes[feat_idx][pairwise[1]]
        yStack.append(Y1)

        yIntersection = np.amin(yStack, axis=0)

        if self.integration_method == "sum":
            OA = (yIntersection.sum())/delta
        elif self.integration_method == "trapz":
            OA = np.trapz(yIntersection, self.XGrid)
        else:
            raise ValueError(
                "Possible options for <integration_method>: " +
                "('trapz', 'sum')"
            )

        return OA

    def construct_kernel(self, feat_idx):
        """
        Construct the kernel density estimator of all the class-segregated groups
        for a given feature.

        Parameters
        ----------
        feat_idx : int
         - Index of the desired feature in the given dataset, X.

        Returns
        -------
        kernels : dict
         - dict[class]: kernel.

        """
        kernels = defaultdict()
        pdes = defaultdict()
        normalizedX_dict = self.normalize_feature_vector(feat_idx)

        for y in self.yLabels:
            kernel = gaussian_kde(normalizedX_dict[y], self.bw_method)
            pde = np.reshape(kernel(self.XGrid).T, self.delta)
            kernels[y] = kernel
            pdes[y] = pde

        if self.mode == "development":
            return kernels, pdes
        elif self.mode == "release":
            return pdes

    def normalize_feature_vector(self, feat_idx):
        """
        Sub-routine to normalize each feature vector, then return a dictionary
        of each normalized feature vector, segregated by class.

        Parameters
        ----------
        feat_idx : int
         - Index of the desired feature in the given dataset, X.

        Returns
        -------
        feature_vector_dict : defaultdict
         - Dictionary containing the normalized feature vector, segregatad per
           class
        """
        # Parameters to "center" the series
        minVal = self.X[:, feat_idx].min()
        maxValCentered = (self.X[:, feat_idx] - minVal).max()

        feature_vector_dict = defaultdict()

        for y in self.yLabels:
            feature_vector_perClass = self.y_segregatedGroup[y][:, feat_idx]

            # Centering the series
            # 1. Subtract series with series.min()
            # 2. Divide series by its max
            normalized_vector = feature_vector_perClass - minVal
            normalized_vector = normalized_vector / maxValCentered

            # If X has a standard deviation of zero, we will perturb just
            # the last element to allow for scipy to carry out a Cholesky
            # Decomposition on the variance matrix
            normalized_vector[-1] += 1e-15 # adding 'zero'

            feature_vector_dict[y] = normalized_vector

        return feature_vector_dict

    def segregateX_y(self):
        """
        Routine to segregate X samples into unique y groups
        """
        unique_y = list(set(self.y))

        _subX = defaultdict()
        for uy in unique_y:
            _subX[uy] = self.X[np.where(self.y==uy)[0], :]

        return _subX

    def get_topnFeatures(self, n):
        """
        Returns the indices of the top n features (smaller intersection areas
        are more important).

        Parameters
        ----------
        n : int
         - Desired number of top features

        Returns
        -------
        inds_topFeatures : list
         - List of top n features, starting from the most to least important
           features.
        """
        return sorted(
            range(len(self.intersectionAreas)),
            key=lambda i: self.intersectionAreas[i],
            reverse=False
        )[:n]

    def plot_overlapAreas(
            self, feat_idx, feat_names=None, _ylim=None, _title=None,
            show_samples=False, savefig=None, _format="svg",
            legend=False, return_normVector=False, _ax=None
    ):
        """
        Function to plot intersection areas for a given feature.

        Parameters
        ----------
        legend : bool, 'intersection'
         - If true, legend would include all the class PDEs and computed
           intersection areas. If 'intersection', only includes
           intersection area.

        savefig : str, None
         - If str, then figure will be saved as file name given.
        """
        normalizedX_dict = self.normalize_feature_vector(feat_idx)

        OA = self.intersectionAreas[feat_idx]

        yStack = []

        if _ax is None:
            fig, _ax = plt.subplots(1,1)
            _ax_passed = False
        else:
            _ax_passed = True

        linecolors = []
        for y, p_y in self.pdes[feat_idx].items():
            yStack.append(p_y)

            # Plotting the probabilty density estimate per class
            if legend == "intersection":
                p = _ax.plot(self.XGrid, p_y, alpha=0.7)
            else:
                if legend:
                    p = _ax.plot(self.XGrid, p_y, alpha=0.7, label=y)
                else:
                    p = _ax.plot(self.XGrid, p_y, alpha=0.7)

            # Get line colors
            linecolors.append(p[0].get_color())

        # Plotting the data samples
        if show_samples:
            yMax = _ax.get_ylim()[1]
            for i, k in enumerate(self.pdes[feat_idx].keys()):
                _ax.vlines(
                    normalizedX_dict[k], 0.0, 0.03*yMax,
                    color=linecolors[i], alpha=0.7
                )

        # Getting the smallest probabilities of all the estimates at every
        # grid point
        yIntersection = np.amin(yStack, axis=0)
        if legend == "intersection":
            _label = r"$A_{i} = $"
            _label += str(round(OA, 3))
            fill_poly = _ax.fill_between(
                self.XGrid, 0, yIntersection, label=_label,
                color="lightgray", edgecolor="lavender"
            )
        else:
            fill_poly = _ax.fill_between(
                self.XGrid, 0, yIntersection, color="lightgray", edgecolor="lavender"
            )

        fill_poly.set_hatch('xxx')

        if not feat_names is None:
            _ax.set_xlabel(feat_names[feat_idx], fontsize='large')
        else:
            _ax.set_xlabel(f"Feature {feat_idx}", fontsize='large')

        xrange = self.XGrid.max() - self.XGrid.min()
        _ax.set_xlim(
            (self.XGrid.min()-(xrange*0.05), self.XGrid.max()+(xrange*0.05))
        )
        _ax.set_xticks(
            np.arange(
                self.XGrid.min()-(xrange*0.05), self.XGrid.max()+(xrange*0.05), 0.2
            )
        )
        if not _ylim is None:
            _ax.set_ylim(_ylim)

        if legend:
            _ax.legend(bbox_to_anchor=(0.4,1), loc='upper left', title=_title, fontsize='x-large')

        # If user DOES NOT pass a matplotlib Axes object from the outside
        if not _ax_passed:
            _ax.grid(visible=True, which="major", axis="both")

            if not savefig is None:
                print(f"Plotting {savefig}.{_format} ... ") 
                plt.savefig(f"{savefig}.{_format}", format=_format)

        if return_normVector:
            return normalizedX_dict
