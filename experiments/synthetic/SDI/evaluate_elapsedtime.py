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

plt.rcParams["font.family"] = ["Arial"]
fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True)
upperplt = sns.barplot(
    data=plot_df, x="FS", y="Average Time", hue="# Class", ax=ax_upper
)
ax_upper.set_ylabel("")
lowerplt = sns.barplot(
    data=plot_df, x="FS", y="Average Time", hue="# Class", ax=ax_lower,
    legend=False
)
ax_lower.set_ylabel("")

ax_upper.set_ylim(50, 150) # Upper end
ax_lower.set_ylim(0, 25)   # Lower end

# Hide the spines between ax_2 and ax_1
ax_upper.spines.bottom.set_visible(False)
ax_lower.spines.top.set_visible(False)

ax_upper.tick_params(labeltop=False)

ax_lower.xaxis.tick_bottom()

# Creating cut-out slanted lines
# Idea: Line objects in four corners of the axes

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)], markersize=12,
    linestyle="none", color='k', mec='k', mew=1, clip_on=False
)
ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)

ax_upper.set(ylabel=None)
ax_lower.set(ylabel=None)

plt.xlabel("Feature Selection Method", fontsize="medium")
fig.supylabel("Average Computational Time [s]", fontsize="medium")
fig.suptitle("SDI Datasets", fontsize="large")

fig.tight_layout()
plt.subplots_adjust(left=0.115, top=0.92, hspace=0.075)

plt.show()

sys.exit(0)
