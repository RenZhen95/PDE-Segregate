import pickle
import os, sys
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3 loocv.py <processedDatasets>")
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])

    
