KDE
Simple and nice explaination of how KDE works:
https://www.youtube.com/watch?v=DCgPRaIDYXA

Something more in-depth:
https://seehuhn.github.io/MATH5714M/X01-KDE.html#kernel-density-estimation

From Scott:
 - The histogram might be proved to be an inadmissble (not allowed) estimator, but that 
 theoretical fact should not be taken to suggest histograms should not be used.
 - Some methods that are theoretically superior are almost never used in practice. The reason 
 is that the ordering of algorithms is not absolute, but is dependent not only on the 
 unknown density but also on the sample size.
 - Thus the histogram is generally superior for small samples regardless of its asymptotic 
 properties.
 - However, if the notion that optimality is all important is adopted, then the focus becomes 
 matching the theoretical properties of an estimator to the assumed properties of the density 
 function. Is it a gross inefficiency to use a procedure that requires only two continuous 
 derivatives when the curve in fact has six continuous derivatives?
 -  In practice, when faced with the application of a procedure that requires six derivatives, 
 or some other assumption that cannot be proved in practice, it is more important to be able 
 to recognize the signs of estimator failure than to worry too much about assumptions.
 - The notions of efficiency and admissibility are related to the choice of a criterion, which 
 can only imperfectly measure the quality of a nonparametric estimate. Unlike optimal parametric 
 estimates that are useful for many purposes, nonparametric estimates must be optimized for each 
 application. The extra work is justified by the extra flexibility.
 - As the choice of criterion is imperfect, so then is the notion of a single optimal estimator. 
 This attitude reflects not sloppy thinking, but rather the imperfect relationship between the 
 practical and theoretical aspects of our methods.
 - Visualization is an important component of nonparametric data analysis.
 - Function visualization is a significant component of nonparametric function estimation, and 
 can draw on the relevant literature in the fields of scientific visualization and computer graphics.
 - Non-parametric curves are driven by structure in the data and are broadly applicable
 - Parametric curves rely on model building and prior knowledge of the equations underlying the data
	- Two phases of parametric estimation: specification and estimation
 - An incorrectly specified parametric model has a bias that cannot be removed by large samples alone.
 - The “curse of optimality” is that incorrect application of “optimal” methods is preferred to more 
 general but less-efficient methods.
 - What Pearson failed to argue persuasively is that optimal estimators can become inefficient with 
 only small perturbations in the assumptions underlying the parametric model.
 - The modern emphasis on robust estimation correctly sacrifices a small percentage of parametric 
 optimality in order to achieve greater insensitivity to model misspecification.
 - In the parametric world, an optimal estimator is likely to be optimal for any related purpose.
 - In the nonparametric world, an estimator may be optimal for one purpose and noticeably suboptimal
 for another. This extra work is a price to be paid for working with a more general class of estimators.
 
From Cai, 2024:
 - Probability estimation via kernel approximation is sensitive to the sample size.
  + Counter argument: PDE-Segregate allows for visualization, thus allowing users to make an informed 
  decision

Feature Selection
1. LH-RELIEF: Feature weight estimation for gene selection: a local hyperlinear learning approach
DOI: https://doi.org/10.1186/1471-2105-15-70
APA: Cai, H., Ruan, P., Ng, M., & Akutsu, T. (2014). Feature weight estimation for gene selection: a local hyperlinear learning approach. BMC bioinformatics, 15(1), 1-13.

2. I-RELIEF: Iterative RELIEF for Feature Weighting: Algorithms, Theories, and Applications
DOI: https://doi.org/10.1109/TPAMI.2007.1093
APA: Sun, Y. (2007). Iterative RELIEF for feature weighting: algorithms, theories, and applications. IEEE transactions on pattern analysis and machine intelligence, 29(6), 1035-1051.

3. RELIEF-F: Estimating attributes: Analysis and extensions of RELIEF
DOI: https://doi.org/10.1007/3-540-57868-4_57
APA: Kononenko, I. (1994, April). Estimating attributes: Analysis and extensions of RELIEF. In European conference on machine learning (pp. 171-182). Berlin, Heidelberg: Springer Berlin Heidelberg.

4. MultiSURF: Benchmarking relief-based feature selection methods for bioinformatics data mining
DOI: https://doi.org/10.1016/j.jbi.2018.07.015
APA: Urbanowicz, R. J., Olson, R. S., Schmitt, P., Meeker, M., & Moore, J. H. (2018). Benchmarking relief-based feature selection methods for bioinformatics data mining. Journal of biomedical informatics, 85, 168-188.

5. Random Forests
DOI: https://doi.org/10.1023/A:1010933404324
APA: Breiman, L. (2001). Random forests. Machine learning, 45, 5-32.

6. ANOVA F-statistic: Statistical Methods for Research Workers
APA: Fischer, R. A., Statistical Methods for research workers, (1925).

7. Mutual Information: Estimating mutual information
DOI: https://doi.org/10.1103/PhysRevE.69.066138
APA: Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical review E, 69(6), 066138.

MISTAKE: A new improved filter-based feature selection model for high-dimensional data
https://link.springer.com/article/10.1007/s11227-019-02975-7

Classifiers:
SVM {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
kNN {nNeighbors=(3, 5, 7, 9)}
Naive Bayes 
LDA

Experimental Setup:
1. Data preprocessing:
 - Standardize X element such that columns of X are centered to have mean 0 and scaled to have standard
   deviation 1 (MATLAB: zscore(X))
 - y labels are (somehow) updated according to the following scheme:
   --------------------------------
       function Y = update_label(Y)
       uy = unique(Y);
       miny = min(uy);
       if miny==-1
           Y = (Y+1)/2+1;
       else
           Y = Y-miny+1;
       end
   --------------------------------
 - Genes whose p-values<0.005/0.05 are also further removed

2. Cai selected for top features ranging from 5-30, increments of 1
3. LOOCV (repeated ten times)
   - provides an unbiased estimate of the generalization error
   - 5-fold CV on training data for hyperparameter tuning

------------------------------------------------------------------------------------------------------------------
Writing points
Datasets citations:
 - Microarray datasets: https://doi.org/10.1093/bioinformatics/bti631
 - Cancer-RNA Sequence: https://doi.org/10.24432/C5R88H
 - Person Gait-Data: https://doi.org/10.24432/C5PP68

Filter methods:
RELIEF algorithm
 - considered to be one of the most successful owing to its simplicity and effectiveness (Cai, 2014)
RELIEF-F algorithm
 - tutorial: https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83
MultiSURF
 - checkout:
 1. https://github.com/EpistasisLab/scikit-rebate
 2. https://epistasislab.github.io/scikit-rebate/using/#multisurf_1
Relief-LH (Cai, 2014)
 - authors suggestion for nearest neighbors:
   - 3-5   : small samples
   - 10-20 : largest samples
 - author however suggested a rule of thumb of 7
 
Filter methods select features according to discriminant criteria based on the characteristics of the
data, independent of any classification algorithms (Cai, 2014).

Discriminant criteria commonly used:
 - Entropy measurements
 - Fisher ratio measurements
 - Mutual information measurements
 - RELIEF-based measurements

According to (Benhar, 2022), univariate approaches first rank features individually using some performance
measures where the final feature subset is then determined by establising a threshold value or specifying
the number of features to retain.

Univariate filters are more computationally efficient compared to multivariate filters and thus applied
in many circumstances (Abellana, 2023).

Literature:
Benhar, H., Hosni, M., & Idri, A. (2022). Univariate and Multivariate Filter Feature Selection for Heart Disease Classification. Journal of Information Science & Engineering, 38(4).