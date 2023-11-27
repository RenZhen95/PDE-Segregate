import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def plot_boxPlot(filePath, ax):
    df = pd.read_csv(filePath, index_col=0)
    df_forBoxPlots = pd.DataFrame(np.zeros((192, 3)), columns=["FS", "BalAcc", "Learning Algorithm"])
    df_forBoxPlots = df_forBoxPlots.astype(
        {"Learning Algorithm": "object", "FS": "object"}, copy=True
    )
    row_count = 0
    for idx in df.index:
        for learningAlgo in list(df.columns):
            df_forBoxPlots.loc[row_count, "FS"] = idx.split('-')[0]
            df_forBoxPlots.loc[row_count, "BalAcc"] = df.at[idx, learningAlgo] 
            df_forBoxPlots.loc[row_count, "Learning Algorithm"] = learningAlgo
            row_count += 1

    axBoxPlot = sns.boxplot(
        data=df_forBoxPlots, x="FS", y="BalAcc", hue="Learning Algorithm",
        ax=ax, legend=False
    )
    # axBoxPlot.set_ylabel("Balanced Accuracy")
    # axBoxPlot.set_xlabel(f"Dataset: {f.name.split('_'[0])}")

    return axBoxPlot, df_forBoxPlots

if len(sys.argv) < 2:
    print("Possible usage: python3 plot_boxPlots.py <resultsFolder>")
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])

# 11 benchmark datasets
fig, axs = plt.subplots(4, 3, figsize=(12, 13), sharey=True)

axisCounter = 0
for f in os.scandir(resultsFolder):
    axBoxPlot, df_forBoxPlots = plot_boxPlot(f, axs[int(axisCounter/3), axisCounter%3])
    axisCounter += 1        

# fig.tight_layout()
# plt.show()
sys.exit(0)
