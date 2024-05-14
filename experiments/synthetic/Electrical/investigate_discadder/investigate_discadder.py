import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

sys.path.append(
    os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
    ))
)

from pde_segregate import PDE_Segregate

if len(sys.argv) < 2:
    print(
        "Possible usage: python3 investigate_discadder.py <processedDatasets>"
    )
    sys.exit(1)
else:
    synthetic_datasets_pkl = Path(sys.argv[1])

nRetainedFeatures = 10 # Top 10 % of 100 (Canedo, 2012)

with open(synthetic_datasets_pkl, "rb") as handle:
    synthetic_datasets = pickle.load(handle)

n30_datasets = synthetic_datasets[30]
n50_datasets = synthetic_datasets[50]
n70_datasets = synthetic_datasets[70]

itr = 0
X30 = n30_datasets[itr]['X']
y30 = n30_datasets[itr]['y']

print(X30)
print(y30)

# Proposed algorithm
tPDE_start = process_time()
pdeSegregate = PDE_Segregate(
    integration_method="trapz", delta=500, bw_method="scott",
    n=2, n_jobs=-1, mode="release", lower_end=-1.5, upper_end=2.5
)
pdeSegregate.fit(X30, y30)
tPDE_stop = process_time()
tPDE = tPDE_stop - tPDE_start

print(pdeSegregate.get_topnFeatures(10))
# [18, 45, 5, 2, 93, 0, 1, 3, 4, 21] for -0.5 to 1.5 (default)
# [18, 45, 93, 14, 25, 35, 85, 31, 65, 13] for 0.0 to 1.0
# [2, 5, 4, 0, 1, 3, 45, 93, 18, 21] for -1.0 to 2.0
# [5, 2, 0, 1, 4, 3, 45, 93, 18, 21] for -1.5 to 2.5
# [5, 2, 0, 1, 4, 3, 45, 93, 18, 21] for -2.0 to 4.0
# [5, 2, 0, 1, 3, 4, 45, 93, 18, 21] for -2.5 to 4.5
# [2, 5, 4, 3, 0, 1, 45, 93, 18, 21] for -5.0 to 6.0

Ai = pdeSegregate.intersectionAreas
print(f"Intersection area of feature 0: {Ai[0]}")
print(f"Intersection area of feature 1: {Ai[1]}")
print(f"Intersection area of feature 2: {Ai[2]}")
# print(f"Intersection area of feature 18: {Ai[18]}")
# print(f"Intersection area of feature 45: {Ai[45]}")
# print(f"Intersection area of feature 5: {Ai[5]}")

pdeSegregate.plot_overlapAreas(0)
pdeSegregate.plot_overlapAreas(1)
pdeSegregate.plot_overlapAreas(2)
# pdeSegregate.plot_overlapAreas(18)
# pdeSegregate.plot_overlapAreas(45)
# pdeSegregate.plot_overlapAreas(5)

import matplotlib.pyplot as plt
plt.show()
