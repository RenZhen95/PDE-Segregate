import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# FSS name-mapper
FSS_dict = {
    "RlfF": "RlfF",
    "MSurf": "MSurf",
    "RlfI": "IRlf",
    "RlfLM": "LHRlf",
    "RFGini": "RFGini",
    "MI": "MI",
    "mRMR": "mRMR",
    "FT": "FT",
    "OAtotal": "PDE-S",
    "OApw": "PDE-S*"
}
# Reading results for the following datasets
datasets = [
    ("cns", "CNS"),
    ("lung", "Lung"),
    ("leuk", "Leukemia"),
    ("colon", "Colon"),
    ("pros3", "Prostrate"),
    ("gcm", "GCM"),
    ("dlbcl", "DLBCL"),
    ("pros1", "Prostrate1"),
    ("pros2", "Prostrate2"),
    ("geneExpressionCancerRNA", "Cancer RNA-Gene"),
    ("PersonGaitDataSet", "Person Gait Classification")
]

if len(sys.argv) < 3:
    print("Possible usage: python3 compare_evaluationgrid.py <resultsFolder1> <resultsFolder2")
    sys.exit(1)
else:
    resultsFolder1 = Path(sys.argv[1])
    resultsFolder2 = Path(sys.argv[2])

def read_results(_folder):
    datasetResults_dict = defaultdict()

    for f in os.scandir(_folder):
        ds_name = (f.name).split('_')[0]
        datasetResults_dict[ds_name] = pd.read_csv(f, index_col=0)
    
        datasetResults_dict[ds_name]["FSS"] = [
            "" for i in range(datasetResults_dict[ds_name].shape[0])
        ]
        datasetResults_dict[ds_name]["nThreshold"] = [
            "" for i in range(datasetResults_dict[ds_name].shape[0])
        ]
        for i in datasetResults_dict[ds_name].index:
            fs = i.split('-')[0]
            th = i.split('-')[1]
            if fs in list(FSS_dict.keys()):
                datasetResults_dict[ds_name].at[i, "FSS"] = FSS_dict[fs]
                datasetResults_dict[ds_name].at[i, "nThreshold"] = th
            else:
                datasetResults_dict[ds_name].drop([i], inplace=True)

    return datasetResults_dict

results1 = read_results(resultsFolder1)
results2 = read_results(resultsFolder2)

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "PDE-S", "PDE-S*"
]

for k in datasets:
    print(k)
    df1 = results1[k[0]].round(decimals=3)
    df1_25  = df1[df1["nThreshold"]=="25"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df1_25 = df1_25.set_index("FSS")

    df1_50  = df1[df1["nThreshold"]=="50"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df1_50 = df1_50.set_index("FSS")

    df1_75  = df1[df1["nThreshold"]=="75"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df1_75 = df1_75.set_index("FSS")

    df1_100 = df1[df1["nThreshold"]=="100"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df1_100 = df1_100.set_index("FSS")

    df2 = results2[k[0]].round(decimals=3)
    df2_25  = df2[df2["nThreshold"]=="25"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df2_25 = df2_25.set_index("FSS")

    df2_50  = df2[df2["nThreshold"]=="50"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df2_50 = df2_50.set_index("FSS")

    df2_75  = df2[df2["nThreshold"]=="75"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df2_75 = df2_75.set_index("FSS")

    df2_100 = df2[df2["nThreshold"]=="100"][
        ["kNN", "SVM", "Gaussian-NB", "LDA", "DT", "FSS"]
    ]; df2_100 = df2_100.set_index("FSS")

    # PDE-S
    df_pdes_comparison = pd.DataFrame(
        index=[
            "eval_0-1_25", "eval_-1-2_25",
            "eval_0-1_50", "eval_-1-2_50",
            "eval_0-1_75", "eval_-1-2_75",
            "eval_0-1_100", "eval_-1-2_100"
        ],
        columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
    )
    df_pdes_comparison.loc["eval_0-1_25",:] = df1_25.loc["PDE-S",:]
    df_pdes_comparison.loc["eval_-1-2_25",:] = df2_25.loc["PDE-S",:]
    df_pdes_comparison.loc["eval_0-1_50",:] = df1_50.loc["PDE-S",:]
    df_pdes_comparison.loc["eval_-1-2_50",:] = df2_50.loc["PDE-S",:]
    df_pdes_comparison.loc["eval_0-1_75",:] = df1_75.loc["PDE-S",:]
    df_pdes_comparison.loc["eval_-1-2_75",:] = df2_75.loc["PDE-S",:]
    df_pdes_comparison.loc["eval_0-1_100",:] = df1_100.loc["PDE-S",:]
    df_pdes_comparison.loc["eval_-1-2_100",:] = df2_100.loc["PDE-S",:]

    df_pdes_difference = pd.DataFrame(
        index=["n25", "n50", "n75", "n100"],
        columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
    )
    df_pdes_difference.loc["n25",:] = df2_25.loc["PDE-S",:] - df1_25.loc["PDE-S",:]
    df_pdes_difference.loc["n50",:] = df2_50.loc["PDE-S",:] - df1_50.loc["PDE-S",:]
    df_pdes_difference.loc["n75",:] = df2_75.loc["PDE-S",:] - df1_75.loc["PDE-S",:]
    df_pdes_difference.loc["n100",:] = df2_100.loc["PDE-S",:] - df1_100.loc["PDE-S",:]
    df_pdes_toplot = pd.DataFrame(data=np.zeros((20, 3)), columns=["Diff", "Clf", "nThreshold"])
    i = 0
    for clf in df_pdes_difference.columns:
        df_pdes_toplot.loc[i, "Diff"] = df_pdes_difference.loc["n25", clf]
        df_pdes_toplot.loc[i, "Clf"] = clf
        df_pdes_toplot.loc[i, "nThreshold"] = "n25"
        df_pdes_toplot.loc[i+1, "Diff"] = df_pdes_difference.loc["n50", clf]
        df_pdes_toplot.loc[i+1, "Clf"] = clf
        df_pdes_toplot.loc[i+1, "nThreshold"] = "n50"
        df_pdes_toplot.loc[i+2, "Diff"] = df_pdes_difference.loc["n75", clf]
        df_pdes_toplot.loc[i+2, "Clf"] = clf
        df_pdes_toplot.loc[i+2, "nThreshold"] = "n75"
        df_pdes_toplot.loc[i+3, "Diff"] = df_pdes_difference.loc["n100", clf]
        df_pdes_toplot.loc[i+3, "Clf"] = clf
        df_pdes_toplot.loc[i+3, "nThreshold"] = "n100"
        i += 4

    fig1, ax1 = plt.subplots(1,1)
    pdes_bar = sns.barplot(data=df_pdes_toplot, x="nThreshold", y="Diff", hue="Clf", ax=ax1)
    plt.title(f"PDE-S for {k[0]} dataset")
    plt.savefig(f"pdes_{k[0]}diffplot.png", format="png")

    # PDE-S*
    df_pdesstar_comparison = pd.DataFrame(
        index=[
            "eval_0-1_25", "eval_-1-2_25",
            "eval_0-1_50", "eval_-1-2_50",
            "eval_0-1_75", "eval_-1-2_75",
            "eval_0-1_100", "eval_-1-2_100"
        ],
        columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
    )
    df_pdesstar_comparison.loc["eval_0-1_25",:] = df1_25.loc["PDE-S*",:]
    df_pdesstar_comparison.loc["eval_-1-2_25",:] = df2_25.loc["PDE-S*",:]
    df_pdesstar_comparison.loc["eval_0-1_50",:] = df1_50.loc["PDE-S*",:]
    df_pdesstar_comparison.loc["eval_-1-2_50",:] = df2_50.loc["PDE-S*",:]
    df_pdesstar_comparison.loc["eval_0-1_75",:] = df1_75.loc["PDE-S*",:]
    df_pdesstar_comparison.loc["eval_-1-2_75",:] = df2_75.loc["PDE-S*",:]
    df_pdesstar_comparison.loc["eval_0-1_100",:] = df1_100.loc["PDE-S*",:]
    df_pdesstar_comparison.loc["eval_-1-2_100",:] = df2_100.loc["PDE-S*",:]

    df_pdesstar_difference = pd.DataFrame(                                                          
        index=["n25", "n50", "n75", "n100"],                                                    
        columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
    )                                                                                           
    df_pdesstar_difference.loc["n25",:] = df2_25.loc["PDE-S*",:] - df1_25.loc["PDE-S*",:]             
    df_pdesstar_difference.loc["n50",:] = df2_50.loc["PDE-S*",:] - df1_50.loc["PDE-S*",:]             
    df_pdesstar_difference.loc["n75",:] = df2_75.loc["PDE-S*",:] - df1_75.loc["PDE-S*",:]             
    df_pdesstar_difference.loc["n100",:] = df2_100.loc["PDE-S*",:] - df1_100.loc["PDE-S*",:]          
    df_pdesstar_toplot = pd.DataFrame(data=np.zeros((20, 3)), columns=["Diff", "Clf", "nThreshold"])
    i = 0                                                                                       
    for clf in df_pdes_difference.columns:                                                      
        df_pdesstar_toplot.loc[i, "Diff"] = df_pdesstar_difference.loc["n25", clf]                      
        df_pdesstar_toplot.loc[i, "Clf"] = clf                                                      
        df_pdesstar_toplot.loc[i, "nThreshold"] = "n25"                                             
        df_pdesstar_toplot.loc[i+1, "Diff"] = df_pdesstar_difference.loc["n50", clf]                    
        df_pdesstar_toplot.loc[i+1, "Clf"] = clf                                                    
        df_pdesstar_toplot.loc[i+1, "nThreshold"] = "n50"                                           
        df_pdesstar_toplot.loc[i+2, "Diff"] = df_pdesstar_difference.loc["n75", clf]                    
        df_pdesstar_toplot.loc[i+2, "Clf"] = clf                                                    
        df_pdesstar_toplot.loc[i+2, "nThreshold"] = "n75"                                           
        df_pdesstar_toplot.loc[i+3, "Diff"] = df_pdesstar_difference.loc["n100", clf]                   
        df_pdesstar_toplot.loc[i+3, "Clf"] = clf                                                    
        df_pdesstar_toplot.loc[i+3, "nThreshold"] = "n100"                                          
        i += 4                                                                                  

    fig2, ax2 = plt.subplots(1,1)
    pdesstar_bar = sns.barplot(data=df_pdesstar_toplot, x="nThreshold", y="Diff", hue="Clf", ax=ax2)
    plt.title(f"PDE-S* for {k[0]} dataset")
    plt.savefig(f"pdesstar_{k[0]}diffplot.png", format="png")

sys.exit(0)
