import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import defaultdict

if len(sys.argv) < 2:
    print(
        "Possible usage: python3.11 evaluate_elapsedtime.py <resultsFolder>"
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])

# Getting elapsed timess
elapsedtime_df = pd.read_csv(
    resultsFolder.joinpath("SDIelapsedtimes.csv"), index_col=0
)
average_time_df = elapsedtime_df.groupby(["nClass"]).mean()
average_time_df = average_time_df.drop(columns=["iteration"])

plot_df = pd.DataFrame(
    data=np.zeros((3*9, 3)), columns=["Average Time", "# Class", "FS"]
)
plot_df["FS"] = plot_df["FS"].astype("object")

i = 0
for fs in average_time_df.columns:
    for c in average_time_df.index:
        plot_df.at[i, "Average Time"] = average_time_df.at[c, fs]
        plot_df.at[i, "# Class"] = c
        plot_df.at[i, "FS"] = fs
        i += 1

sns.barplot(
    data=plot_df, x="FS", y="Average Time", hue="# Class"
)
plt.xlabel("Feature Selection Method")
plt.show()

sys.exit(0)
