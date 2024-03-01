import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

class PDE_Segregate():
    def __init__(
            self, X, y, integration_method, delta=1000, bw_method="scott", n_jobs=1
    ):
        self.X = X
        self.y = y

        self.compute_PDEintersectionAreas(
            integration_method, delta, bw_method, n_jobs
        )

    def compute_PDEintersectionAreas(self, integrate, delta, bw_method, n_jobs):
        """
        Only useful for multi-class classification problems.

        Gets the average of the pair-wise intersection areas of the PDE of
        between the different class-segregated groups.
        """
        # Number of different combinations: nclass choose 2
        pairwise_combinations = combinations(self.yLabels, 2)
        
 
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
        return -1*self.intersectionAreas


    def compute_PDEintersectionAreas(self, integrate, delta, bw_method, n_jobs):
        """
        Get the intersection areas of the PDE of class-segregated groups.
        """
        # Grouping the samples according to unique y label
        self.y_segregatedGroup = self.segregateX_y()

        # You can't build a distribution function with only one sample
        yToRemove = []

        for y in self.y_segregatedGroup.keys():
            if self.y_segregatedGroup[y].shape[0] < 2:
                yToRemove.append(y)
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

        self.intersectionAreas = np.zeros((self.X.shape[1],))
        print("Computing the intersection area of each feature ... ")

        # Spreading over multiple threads
        delayed_calls = [
            delayed(
                self.compute_OA
            )(feat_idx, integrate, delta, bw_method) for feat_idx in range(self.X.shape[1])
        ]
        self.intersectionAreas = Parallel(n_jobs=n_jobs)(delayed_calls)
        
        # for feat_idx in tqdm(range(self.X.shape[1])):
        #     OA, kernels, lengths = self.compute_OA(
        #         feat_idx, integrate, delta, bw_method
        #     )
        #     self.intersectionAreas[feat_idx] = OA

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
        res_tmp = self.intersectionAreas
        res_tmpSorted = np.sort(res_tmp)

        inds_topFeatures = []

        counter = 1

        OAs = []

        for i in res_tmpSorted:
            OAs.append(i)
            if i in OAs:
                pass

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

    def compute_OA(self, feat_idx, integrate, delta, bw_method, only_areas=True, return_series=False):
        """
        Compute the intersection areas of the PDE of class-segregated groups
        for a given feature.

        Parameters
        ----------
        feat_idx : int
         - Index of the desired feature in the given dataset, X.

        delta : int
         - Number of cells in the x-grid

        bw_method : str, scalar or callable
         - The method used to calculate the estimator bandwith. This can be
           'scott' and 'silverman', a scalar constant or a callable. For
           more details, see scipy.stats.gaussian_kde documentation.

        only_areas : bool
         - Option to only return the computed intersection areas.

        reture_series : bool
         - Option to return the centered series

        Returns
        -------
        OA : float
         - Computed intersection region of the PDEs.

        kernels : list
         - List of kernels containing kernel density estimator of each class.

        lengths : list
         - List of sample size of each class.

        normalizedX : numpy.array
         - Array of normalized feature vector.
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

        normalizedX = defaultdict()

        for y in self.yLabels:
            X_feat_idx = self.y_segregatedGroup[y][:, feat_idx]

            # Centering the series
            # 1. Subtract series with series.min()
            # 2. Divide series by its max
            X_feat_idxNormalized = X_feat_idx - minVal
            X_feat_idxNormalized = X_feat_idxNormalized / maxValCentered

            lengths.append(X_feat_idxNormalized.shape[0])

            # If X has a standard deviation of zero, we will perturb just
            # the last element to allow for scipy to carry out a Cholesky
            # Decomposition on the variance matrix
            X_feat_idxNormalized[-1] += 1e-15 # adding 'zero'

            kernel = gaussian_kde(X_feat_idxNormalized, bw_method)
            kernels.append((y, kernel))

            if return_series:
                normalizedX[y] = X_feat_idxNormalized

        # There must be AT LEAST two target groups with non-zero S.D, or this feature
        # ranking method just makes no sense
        if len(lengths) < 2:
            print(
                f"\nThe feature (idx:{feat_idx}) has LESS than two feature groups with " +
                "non-zero S.D and thus not suitable for this feature selection method. " +
                "Intersection area of PDE will simply be assigned a value of 10.0"
            )
            OA = 10.0
            time.sleep(10)
        else:
            # Initializing the x-axis grid
            print(f"Delta: {delta}")
            XGrid = np.linspace(0, 1, delta)

            yStack = []
            for k in kernels:
                Y = np.reshape(k[1](XGrid).T, delta)
                yStack.append(Y)

            yIntersection = np.amin(yStack, axis=0)

            if integrate == "sum":
                OA = (yIntersection.sum())/delta
            elif integrate == "trapz":
                OA = np.trapz(yIntersection, XGrid)

        if only_areas:
            return OA
        else:
            if return_series:
                return OA, kernels, lengths, normalizedX
            else:
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
            self, feat_idx, feat_names=None, _ylim=None, _title=None,
            show_samples=False, savefig=None, format="svg",
            legend=False, _ax=None
    ):
        """
        Function to plot intersection areas for a given feature.
        """
        OA, _kernels, _lengths, normalizedX = self.compute_OA(
            feat_idx, only_areas=False, return_series=True
        )

        yStack = []
        _xGrid = np.linspace(0, 1, max(1000, max(_lengths)))

        if _ax is None:
            fig, _ax = plt.subplots(1,1)

        linecolors = []
        for k in _kernels:
            Y = np.reshape(k[1](_xGrid).T, _xGrid.shape)
            yStack.append(Y)

            # Plotting the probabilty density estimate per class
            p = _ax.plot(_xGrid, Y, label=k[0])
            # Get line colors
            linecolors.append(p[0].get_color())

        # Plotting the data samples
        if show_samples:
            yMax = _ax.get_ylim()[1]
            for i, k in enumerate(_kernels):
                _ax.vlines(
                    normalizedX[k[0]], 0.0, 0.03*yMax,
                    color=linecolors[i], alpha=0.7
                )

        # Getting the smallest probabilities of all the estimates at every
        # grid point
        yIntersection = np.amin(yStack, axis=0)
        if legend == "intersection":
            intersectionLegend = True
            fill_poly = _ax.fill_between(
                _xGrid, 0, yIntersection, label=f'Intersection: {round(OA, 3)}',
                color="lightgray", edgecolor="lavender"
            )
        else:
            intersectionLegend = False
            fill_poly = _ax.fill_between(
                _xGrid, 0, yIntersection, color="lightgray", edgecolor="lavender"
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
