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
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    )
)

from pde_segregate import PDE_Segregate

if len(sys.argv) < 4:
    print(
        "Possible usage: python3.11 investigate_cancerRNA.py <processedDatasets> " +
        "<featureRanks> <featureRanksOld>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])
    topranks = Path(sys.argv[2])
    topranks_old = Path(sys.argv[3])

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)

with open(topranks, "rb") as handle:
    topranks = pickle.load(handle)

with open(topranks_old, "rb") as handle:
    topranksOld = pickle.load(handle)

geneRNA_ranks = topranks["geneExpressionCancerRNA"]
geneRNA_ranksOld = topranksOld["geneExpressionCancerRNA"]

# print(geneRNA_ranks["MSurf"])
# print(geneRNA_ranksOld["MSurf"])

X = datasets_dict["geneExpressionCancerRNA"]['X']
y = datasets_dict["geneExpressionCancerRNA"]['y']

# # === === === === === === === === === === === === === === === === === === === ===
tstart = process_time()
pdeSegregate= PDE_Segregate(
    integration_method="trapz", delta=1500, bw_method="scott", n=2, n_jobs=-1,
    mode="development"
)
pdeSegregate.fit(X, y)
tstop = process_time()
t = tstop - tstart
print(t)

inds_topFeatures = pdeSegregate.get_topnFeatures(3)
print(inds_topFeatures)

pdeSegregate.plot_overlapAreas(16920, legend=True, savefig=f"idx16920new")
pdeSegregate.plot_overlapAreas(12537, legend=True, savefig=f"idx12537new")
pdeSegregate.plot_overlapAreas(14792, legend=True, savefig=f"idx14792new")

pdeSegregate.plot_overlapAreas(1842, legend=True, savefig=f"idx1842new")
pdeSegregate.plot_overlapAreas(3525, legend=True, savefig=f"idx3525new")
pdeSegregate.plot_overlapAreas(4368, legend=True, savefig=f"idx4368new")

pdeSegregate.plot_overlapAreas(17547, legend=True, savefig=f"msurf_idx17547new")
pdeSegregate.plot_overlapAreas(18492, legend=True, savefig=f"msurf_idx18492new")
pdeSegregate.plot_overlapAreas(7949, legend=True, savefig=f"msurf_idx7949new")

with open("pdes_new_Ai.pkl", "wb") as handle:
    pickle.dump(pdeSegregate.intersectionAreas, handle)

with open("pdes_new_kernels.pkl", "wb") as handle:
    pickle.dump(pdeSegregate.feature_kernels, handle)
# # === === === === === === === === === === === === === === === === === === === ===

# Old 0.0 - 1.0  : [16920, 12537, 14792]
# New -1.0 - 2.0 : [1842, 3525, 4368]
# MSurf          : [17547, 18492, 7949]

# # === === === === === === === === === === === === === === === === === === === ===
# with open("pdes_new_Ai.pkl", "rb") as handle:
#     newAi = pickle.load(handle)

# print(newAi[1842])
# print(newAi[3525])
# print(newAi[4368])

# print(newAi[16920])
# print(newAi[12537])
# print(newAi[14792])

# print(newAi[17547])
# print(newAi[18492])
# print(newAi[7949])

# print("------------------------------------")
# with open("pdes_old_Ai.pkl", "rb") as handle:
#     oldAi = pickle.load(handle)

# print(oldAi[1842])
# print(oldAi[3525])
# print(oldAi[4368])

# print(oldAi[16920])
# print(oldAi[12537])
# print(oldAi[14792])

# print(newAi[17547])
# print(newAi[18492])
# print(newAi[7949])
