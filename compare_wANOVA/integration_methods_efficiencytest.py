import pickle
import os, sys
import numpy as np
import pandas as pd
from time import process_time

sys.path.append(os.path.dirname(os.getcwd()))
from pde_segregate import PDE_Segregate

# Creating the Generator instance
rng = np.random.default_rng(122807528840384100672342137672332424406)
print(rng.random()) # as check

deltas = [
    100, 250, 500, 750, 1000, 1500, 2000,
    2500, 3000, 3500, 4000, 4500, 5000
]
n = 3000
n0 = n
n1 = n
separation_factor = 1
class0_sd = 1.0
class1_sd = 1.0

# Class 0
class0 = rng.normal(loc=0.0, scale=class0_sd, size=n0)
class0_calcsd = np.std(class0, ddof=1)

# Class 1
class1 = rng.normal(
    loc=separation_factor*class0_calcsd, scale=class1_sd, size=n1
)

# Initializing the dataset
X = pd.DataFrame(
    data=np.zeros((n0+n1, 2)), columns=["Feature", "Class"]
)
X.iloc[:n0, 0] = class0

X.iloc[n0:, 0] = class1
X.iloc[n0:, 1] = 1

y = X["Class"]
X.drop(columns=["Class"], inplace=True)
X = X.values

elapsedTimeDF = pd.DataFrame(
    data=np.zeros((len(deltas), 4)), index=deltas,
    columns=["t_sum", "A_sum", "t_trapz", "A_trapz"]
)

for _delta in deltas:
    for int_method in ["sum", "trapz"]:
        # Computing 1000 times to see the elapsed time
        areas = np.zeros((1000,1))
        t_start = process_time()
        for i in range(1000):
            pdeSegregate = PDE_Segregate(X, y, int_method, delta=_delta)
            areas[i] = pdeSegregate.overlappingAreas[0]
        t_stop = process_time()
        t = t_stop - t_start

        elapsedTimeDF.at[_delta, f"t_{int_method}"] = t
        elapsedTimeDF.at[_delta, f"A_{int_method}"] = areas.mean()

print(elapsedTimeDF)
elapsedTimeDF.to_csv("elapsedTimeDF.csv")

sys.exit()
    
