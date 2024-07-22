To reconstruct results from the paper (liaw2024) run the scripts in the following order as shown below.

**Note**: 
- The following feature selection methods are implemented in Matlab:
  - IRelief [(Sun et al., 2010)][IRlf]
  - LHR [(Cai et al., 2014)][LHR]
  - mRMR [(Ding and Peng, 2003)][mRMR]

- All the scripts require modifications to the paths of corresponding input files, either as input arguments or hard-coded in the scripts 

## Results from synthetic dataset
### Electrical datasets
**Step 0**

In the working directory, create the following subfolders:
-  PDE-S
-  OtherFS
-  IRelief
-  LHRelief
-  mRMR
-  Combined

**Step 1**

Generate the datasets by runing the script (code adapted from paper by [Kamalov et al., 2022][1]):
```
python3 synthetic/Electrical/generate_artificialdatasets.py
```

This will produce four Python-pickled binary files:
- ANDOR continuous datasets
- ADDER continuous datasets
- ANDOR discrete datasets
- ADDER discrete datasets

**Step 2**

Carry out feature selection with PDE-Segregate:
```  
python3 synthetic/Electrical/PDE-S_Electrical.py
```

**Step 3**

Carry out feature selection with other features selection methods:
```  
python3 synthetic/Electrical/featureSelection_Electrical.py
```
and methods implemented in Matlab:
- synthetic/matlabFS/featureSelection_Electrical.m (I-Relief and LHR)
- synthetic/matlabFS/featureSelection_mRMR_Electrical.m (mRMR)

**Step 4**

Carry out preprocessing step to combine feature selection results from the different scripts
```
python3 synthetic/Electrical/combine_fss.py
```

**Step 5**

Evaluations
- Success rates according to [Bolón-Canedo et al. (2013)][SucRate]
  ```
  python3 synthetic/Electrical/evaluate_fss.py
  ```
- Computational time
  ```
  python3 synthetic/Electrical/evaluate_elapsedtime.py
  ```
- Classification accuracy via 10-fold stratified cross-validation
  ```
  python3 synthetic/Electrical/10stratifiedcv.py
  ```

---
[1]: <https://doi.org/10.48550/arXiv.2211.03035>
[IRlf]: <https://doi.org/10.1109/TPAMI.2009.190>
[LHR]: <https://doi.org/10.1186/1471-2105-15-70>
[mRMR]: <https://doi.org/10.1109/CSB.2003.1227396>
[SucRate]: <https://doi.org/10.1007/s10115-012-0487-8>
