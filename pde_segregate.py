import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

class PDE_Segregate():
    def __init__(
            self, X, y, integration_method="trapz", delta=500, bw_method="scott",
            pairwise=False, n_jobs=1
    ):
        """
        Parameters
        ----------
        X : np.array
         - Dataset with the shape: (n_samples, n_features)

        y : np.array
         - Class vector

        integration_method : str
         - Integration method.

           Available options include 'numpy.trapz' (default) and 'sum'.

        delta : int
         - Number of cells in the x-grid

        bw_method : str, scalar or callable
         - The method used to calculate the estimator bandwith. This can be
           'scott' and 'silverman', a scalar constant or a callable. For
           more details, see scipy.stats.gaussian_kde documentation.

        pairwise : bool
         - Option to either compute the intersection area of all the different
           classes OR the mean of pairwise intersection areas.

        n_jobs : int
         - Number of processors to use. -1 to use all available processors.
        """
        self.X = X
        self.y = y
        self.integration_method = integration_method
        self.delta = delta
        self.bw_method = bw_method
        self.pairwise = pairwise
        self.n_jobs = n_jobs

        # Initializing the x-axis grid
        self.XGrid = np.linspace(0, 1, self.delta)

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

    def fit(self):
        """
        Get the intersection areas of the PDE of class-segregated groups.
        """
        # Construct kerndel density estimator per class for every feature
        delayed_calls = (
            delayed(
                self.construct_kernel
            )(feat_idx) for feat_idx in range(self.X.shape[1])
        )
        self.feature_kernels = Parallel(n_jobs=self.n_jobs)(delayed_calls)
        print("Kernels constructed ... ")

        # Computing the intersection area among all classes
        if not self.pairwise:
            delayed_calls_intersectionArea = (
                delayed(
                    self.compute_intersectionArea
                )(feat_idx, self.pairwise) for feat_idx in range(self.X.shape[1])
            )
            intersectionAreas = Parallel(
                n_jobs=self.n_jobs, backend="threading", verbose=1
            )(delayed_calls_intersectionArea)

            self.intersectionAreas = np.array(intersectionAreas)

        # Computing pairwise intersection areas
        else:
            # Number of different combinations: nclass choose 2
            pairwise_combinations = combinations(self.yLabels, 2)
            cStack = []
            for c in pairwise_combinations:
                delayed_calls_pairwiseIntersectionArea = (
                    delayed(
                        self.compute_intersectionArea
                    )(feat_idx, c) for feat_idx in range(self.X.shape[1])
                )
                c_intersection = Parallel(
                    n_jobs=self.n_jobs, backend="threading", verbose=1
                )(delayed_calls_pairwiseIntersectionArea)

                cStack.append(c_intersection)

            cStack = np.array(cStack)
            self.intersectionAreas = np.mean(cStack, axis=0)

    def compute_intersectionArea(self, feat_idx, pairwise):
        """
        Compute intersection area between estimated PDEs.

        Parameters
        ----------
        feat_idx : int
         - Index of the desired feature in the given dataset, X.

        pairwise : False or tuple
         - If false, then compute intersection area of all classes, else a
        pair of indices indicating which pair of classes to compare.

        Returns
        -------
        OA : float
         - Computed intersection area of the PDEs.
        """
        yStack = []

        if not pairwise:
            for Y in self.feature_kernels[feat_idx].values():
                yStack.append(Y)
        else:
            Y0 = self.feature_kernels[feat_idx][pairwise[0]]
            yStack.append(Y0)

            Y1 = self.feature_kernels[feat_idx][pairwise[1]]
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
        normalizedX_dict = self.normalize_feature_vector(feat_idx)

        for y in self.yLabels:
            kernel = gaussian_kde(normalizedX_dict[y], self.bw_method)
            pde = np.reshape(kernel(self.XGrid).T, self.delta)
            kernels[y] = pde

        return kernels

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

    def get_scores(self):
        """
        Get feature importance based on feature's ability to segregate the
        PDEs of the class-segregated samples.c Returns the negative intersection
        areas of the PDEs, such that the lower the score (i.e. the more
        negative the area), the less important the feature.

        Returns
        -------
        numpy.array
         - The NEGATIVE of the computed integral quantifying the intersection
           of the areas below all the probability density estimate curves of
           each feature.
        """
        return -1 * self.intersectionAreas

    def get_topnFeatures(self, n):
        """
        Returns the indices of the top n features.

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
        res_tmp = self.intersectionAreas.copy()
        res_tmpSorted = np.sort(res_tmp) # sorts smallest to largest

        inds_topFeatures = []

        counter = 1

        for i in res_tmpSorted:
            sel_inds = np.where(res_tmp==i)[0]

            # Dealing with intersection areas, which multiple features
            # share (e.g. 0.0)
            if len(sel_inds) == 1:
                inds_topFeatures.append(sel_inds[0])
                counter += 1
            else:
                print(
                    f"The following {len(sel_inds)} features:\n{sel_inds}" +
                    "\nhave the same computed intersection intersection areas: " +
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

    def plot_overlapAreas(
            self, feat_idx, feat_names=None, _ylim=None, _title=None,
            show_samples=False, savefig=None, format="svg",
            legend=False, _ax=None
    ):
        """
        Function to plot intersection areas for a given feature.
        """
        normalizedX_dict = self.normalize_feature_vector(feat_idx)

        OA = self.compute_intersectionArea(feat_idx, False)

        yStack = []

        if _ax is None:
            fig, _ax = plt.subplots(1,1)

        linecolors = []
        for y, p_y in self.feature_kernels[feat_idx].items():
            yStack.append(p_y)

            # Plotting the probabilty density estimate per class
            p = _ax.plot(self.XGrid, p_y, label=y)
            # Get line colors
            linecolors.append(p[0].get_color())

        # Plotting the data samples
        if show_samples:
            yMax = _ax.get_ylim()[1]
            for i, k in enumerate(self.feature_kernels[feat_idx].keys()):
                _ax.vlines(
                    normalizedX_dict[k], 0.0, 0.03*yMax,
                    color=linecolors[i], alpha=0.7
                )

        # Getting the smallest probabilities of all the estimates at every
        # grid point
        yIntersection = np.amin(yStack, axis=0)
        if legend == "intersection":
            intersectionLegend = True
            fill_poly = _ax.fill_between(
                self.XGrid, 0, yIntersection, label=f'Intersection: {round(OA, 3)}',
                color="lightgray", edgecolor="lavender"
            )
        else:
            intersectionLegend = False
            fill_poly = _ax.fill_between(
                self.XGrid, 0, yIntersection, color="lightgray", edgecolor="lavender"
            )

        fill_poly.set_hatch('xxx')

        if not feat_names is None:
            _ax.set_xlabel(feat_names[feat_idx])
        else:
            _ax.set_xlabel(f"Feature {feat_idx}")

        _ax.set_xlim((-0.005, 1.005))
        _ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        if not _ylim is None:
            _ax.set_ylim(_ylim)

        if legend:
            _ax.legend(bbox_to_anchor=(0.8,1), loc='upper left', title=_title, fontsize='small')

        # If user DOES NOT pass a matplotlib Axes object from the outside
        if _ax is None:
            _ax.grid(visible=True, which="major", axis="both")

            if not savefig is None:
                fig.savefig(f"{savefig}.{format}", format=format)
            else:
                plt.show()
