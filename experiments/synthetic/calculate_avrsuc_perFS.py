import os, sys
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3.11 calculate_avrsuc_perFS.py <folder>")
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

electricalFolder = folder.joinpath("Electrical/Results")

ANDORdiscrete = pd.read_csv(
    electricalFolder.joinpath("ANDORdiscrete_successrates.csv"),
    index_col=0
)
ANDORcontinuous = pd.read_csv(
    electricalFolder.joinpath("ANDORcontinuous_successrates.csv"),
    index_col=0
)

ADDERdiscrete = pd.read_csv(
    electricalFolder.joinpath("ADDERdiscrete_successrates.csv"),
    index_col=0
)
ADDERcontinuous = pd.read_csv(
    electricalFolder.joinpath("ADDERcontinuous_successrates.csv"),
    index_col=0
)

SDFolder = folder.joinpath("SDI/Results")
SDI = pd.read_csv(SDFolder.joinpath("20_SDIsuccessrates.csv"), index_col=0)

success_rates = pd.concat(
    [
        ANDORdiscrete, ANDORcontinuous,
        ADDERdiscrete, ADDERcontinuous,
        SDI
    ], ignore_index=True
)

print(success_rates)
print(success_rates.mean())
