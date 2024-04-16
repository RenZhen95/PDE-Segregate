import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

if len(sys.argv) < 3:
    print(
        "Possible usage: python3.11 evaluate_elapsedtime.py <resultsFolder> " +
        "<datasetName>"
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])
    datasetName = sys.argv[2]

# Getting elapsed timess
elapsedtime_df = pd.read_csv(
    resultsFolder.joinpath(f"{datasetName}_elapsedtimes.csv"), index_col=0
)
average_time_df = elapsedtime_df.groupby(["n_obs"]).mean()
average_time_df = average_time_df.drop(columns=["iteration"])

plot_df = pd.DataFrame(
    data=np.zeros((3*9, 3)), columns=["Average Time", "# Observations", "FS"]
)
plot_df["FS"] = plot_df["FS"].astype("object")

i = 0
for fs in average_time_df.columns:
    for n in average_time_df.index:
        plot_df.at[i, "Average Time"] = average_time_df.at[n, fs]
        plot_df.at[i, "# Observations"] = n
        plot_df.at[i, "FS"] = fs
        i += 1

sns.barplot(
    data=plot_df, x="FS", y="Average Time", hue="# Observations"
)
plt.xlabel("Feature Selection Method")
plt.show()

sys.exit(0)
