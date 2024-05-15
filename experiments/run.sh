#/usr/bin/env bash

python3.11 synthetic/Electrical/PDE-S_Electrical.py ~/repo/PDE-SegregateDatasets/synthetic/Electrical/ANDORdiscrete_datasets.pkl ANDORdiscrete
python3.11 synthetic/Electrical/PDE-S_Electrical.py ~/repo/PDE-SegregateDatasets/synthetic/Electrical/ANDORcontinuous_datasets.pkl ANDORcontinuous
python3.11 synthetic/Electrical/PDE-S_Electrical.py ~/repo/PDE-SegregateDatasets/synthetic/Electrical/ADDERdiscrete_datasets.pkl ADDERdiscrete
python3.11 synthetic/Electrical/PDE-S_Electrical.py ~/repo/PDE-SegregateDatasets/synthetic/Electrical/ADDERcontinuous_datasets.pkl ADDERcontinuous

python3.11 synthetic/SDI/PDE-S_SDI.py ~/repo/PDE-SegregateDatasets/synthetic/SDI/

python3.11 synthetic/NSL-KDD/PDE-S_NSLKDD.py ~/repo/PDE-SegregateDatasets/synthetic/NSL-KDD/ProcessedCSV/

python3.11 real/PDE-S_real.py ~/repo/PDE-SegregateDatasets/real/processedDatasets_noANOVAcutclean.pkl 100 ~/repo/PDE-SegregateDatasets/real/Results_052024_1p5/
