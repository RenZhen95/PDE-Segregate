import pickle

with open("pdes_latest05_Ai.pkl", "rb") as handle:
    Ai = pickle.load(handle)

print(Ai)
