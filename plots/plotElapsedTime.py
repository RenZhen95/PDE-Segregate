import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

if len(sys.argv) < 4:
    print("Possible usage: python3 plotElapsedTime.py <pyTime> <IRlf> <LHRlf>")
    sys.exit()
else:
    pytime = Path(sys.argv[1])
    IR_time = Path(sys.argv[2])
    LHR_time = Path(sys.argv[3])

with open(pytime, "rb") as handle:
    pytimes = pickle.load(handle)

IRlf = pd.read_csv(IR_time, header=None, index_col=0)
LHRlf = pd.read_csv(LHR_time, header=None, index_col=0)

FSS = {
    "RlfF": "RELIEF-F",
    "MSurf": "MultiSURF",
    "I-RELIEF": "I-RELIEF",
    "LH-RELIEF": "LH-RELIEF",
    "MI": "Mutual Information",
    "RFGini": "Random Forest (Gini)",
    "FT": "ANOVA (F-Statistic)",
    "OAtotal": "PDE-Segregate"
}

datasets = [
    "cns", "lung", "leuk",
    "dlbcl", "pros3", "geneExpressionCancerRNA",
    "colon", "gcm"
]

elapsedTimes_df = pd.DataFrame(
    data=np.zeros((len(datasets), len(FSS))),
    index=datasets, columns=FSS
)

for fss in FSS.keys():
    for ds in datasets:
        if fss == "I-RELIEF":
            elapsedTimes_df.at[ds, fss] = IRlf.at[ds, 1]
        elif fss == "LH-RELIEF":
            elapsedTimes_df.at[ds, fss] = LHRlf.at[ds, 1]
        else:
            elapsedTimes_df.at[ds, fss] = pytimes[ds][fss]

averageElapsedTime = elapsedTimes_df.mean(axis=0)

averagedTimes_row = pd.DataFrame(
    data=np.reshape(averageElapsedTime.values, (1, elapsedTimes_df.shape[1])),
    columns=averageElapsedTime.index, index=["Average"]
)

elapsedTimes_df = elapsedTimes_df._append(averagedTimes_row)
print(elapsedTimes_df)

df_for_plotting = pd.DataFrame(
    data=np.zeros((elapsedTimes_df.shape[0]*elapsedTimes_df.shape[1], 3)),
    columns=["FSS", "Dataset", "Time [s]"]
)
i = 0
for fss in elapsedTimes_df.columns:
    for ds in datasets:
        df_for_plotting.at[i, "FSS"] = FSS[fss]
        df_for_plotting.at[i, "Dataset"] = ds
        df_for_plotting.at[i, "Time [s]"] = elapsedTimes_df.at[ds, fss]
        i += 1

    df_for_plotting.at[i, "FSS"] = FSS[fss]
    df_for_plotting.at[i, "Dataset"] = "Average"
    df_for_plotting.at[i, "Time [s]"] = elapsedTimes_df.at["Average", fss]
    i += 1

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
axBoxPlot = sns.boxplot(
    data=df_for_plotting, x="Time [s]", y="FSS",
    width=0.7, fliersize=0., fill=False, ax=ax[0]
)
ax[0].set_xlim((0.0, 4500.0))
ax[0].set_ylabel("")
ax[0].set_title(
    "Average Elapsed Times per Feature Selection Method", fontsize="xx-large",
    x=0.0, y=1., horizontalalignment="left"
)
ax[0].tick_params(axis="both", labelsize="large")
ax[0].set_xlabel("CPU Time [s]", fontsize="large")

df_for_plotting_reduced = df_for_plotting[df_for_plotting["FSS"] != "I-RELIEF"]
df_for_plotting_reduced = df_for_plotting_reduced[
    df_for_plotting_reduced["FSS"] != "LH-RELIEF"
]

axBoxPlot2 = sns.boxplot(
    data=df_for_plotting_reduced, x="Time [s]", y="FSS",
    width=0.7, fliersize=0., fill=False, ax=ax[1]
)
ax[1].set_xlim((0.0, 120.0))
ax[1].set_ylabel("")
ax[1].tick_params(axis="both", labelsize="large")
ax[1].set_xlabel("CPU Time [s]", fontsize="large")

plt.tight_layout()
plt.show()

sys.exit()
