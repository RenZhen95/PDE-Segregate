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

Feature Selection
1. A new improved filter-based feature selection model for high-dimensional data
https://link.springer.com/article/10.1007/s11227-019-02975-7
(mainly for the microarray datasets)
2. Feature weight estimation for gene selection: a local hyperlinear learning approach
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-70#Sec2

Classifiers:
SVM {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
kNN {nNeighbors=(3, 5, 7, 9)}
Naive Bayes 
LDA

TODO:
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
 - Genes whose p-values<0.005 are also further removed

2. Cai selected for top features ranging from 5-30, increments of 1
3. LOOCV (repeated ten times)
   - provides an unbiased estimate of the generalization error
   - 5-fold CV on training data for hyperparameter tuning

------------------------------------------------------------------------------------------------------------------
Writing points
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


