import os, sys
import scipy.io
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(1)
else:
    datasetPath = Path(sys.argv[1])

dataObject = scipy.io.loadmat(datasetPath)
data = dataObject["data"]

df = pd.DataFrame(data)
print(df)
print(df.shape)
print(df.summary())
