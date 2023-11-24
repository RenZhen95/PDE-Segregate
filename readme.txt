KDE
Simple and nice explaination of how KDE works:
https://www.youtube.com/watch?v=DCgPRaIDYXA

Something more in-depth:
https://seehuhn.github.io/MATH5714M/X01-KDE.html#kernel-density-estimation

Feature Selection
1. A new improved filter-based feature selection model for high-dimensional data
https://link.springer.com/article/10.1007/s11227-019-02975-7
(mainly for the microarray datasets)
2. Feature weight estimation for gene selection: a local hyperlinear learning approach
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-70#Sec2

Treating the trimming threshold as a part of the hyperparameter tuning
Krstajic, 2014

Classifiers:
SVM
kNN
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
   
Filter methods:
RELIEF algorithm
 - considered to be one of the most successful owing to its simplicity and effectiveness (Cai, 2014)
RELIEF-F algorithm
 - tutorial: https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83
MultiSURF
 - checkout:
 1. https://github.com/EpistasisLab/scikit-rebate
 2. https://epistasislab.github.io/scikit-rebate/using/#multisurf_1

Filter methods select features according to discriminant criteria based on the characteristics of the
data, independent of any classification algorithms (Cai, 2014).

Discriminant criteria commonly used:
 - Entropy measurements
 - Fisher ratio measurements
 - Mutual information measurements
 - RELIEF-based measurements


