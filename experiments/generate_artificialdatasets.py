import pickle
import random
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy import logical_or as lor
from numpy import logical_and as land
from numpy import logical_not as lnot
from numpy import logical_xor as lxor

# Generate a list of all possible combinations of 3-bits
def gen_3():
  rlvnt = []
  for i in [0,1]:
    for j in [0,1]:
      for k in [0,1]:
        rlvnt.append([i,j,k])
  return rlvnt

# Create 2 correlated features in case of binary target (y)
# by randomly fliping 30% of the values of y
def make_cor(y):
  random.seed(0)
  cor_vars = []
  for i in range(2):
    cor_i = y.copy()
    ind = random.sample(range(len(y)), int(0.3*len(y)))
    cor_i[ind] = lnot(cor_i[ind])
    cor_vars.append(cor_i)
  return np.array(cor_vars).transpose()

# Generate a list of all possible combinations of 3-bits
def gen_4():
  rlvnt_0 = gen_3()
  for seq in rlvnt_0:
    seq.append(0)

  rlvnt_1 = gen_3()
  for seq in rlvnt_1:
    seq.append(1)

  return rlvnt_0 + rlvnt_1

def reconstruct_continuousVariables(x, seed=0):
    """
    Function to 'reconstruct' continuous variables based on a given
    binary feature array.
    """
    # Construct random noise then scaling it to 0-1
    rng = np.random.default_rng(seed=seed)
    noise = rng.standard_normal(
        size=(x.shape[0], x.shape[1])
    )

    # Added 1e-3 in denominator so that maximum bound is slightly
    # below 1, to achieve val+noise < 1 equals 0
    # (in that way, features with value 0 will NEVER get a noise of 1)
    noise_scaled = (
        noise - noise.min(axis=0)
    ) / (noise.max(axis=0) - noise.min(axis=0) + 1e-3)

    return x + noise_scaled

# ANDOR dataset + Comparator
def andor(n_obs=50, n_I=90, seed=0):
    np.random.seed(seed)

    # Redundant features (negation of relevant features)
    red = lnot(gen_4()).astype(int)

    # Relevant + redundant features
    rr = np.hstack([gen_4(), red])

    # Quotient of n_obs divided by 16
    q = n_obs // 16

    # Remainder of n_obs divided by 16
    r = n_obs % 16

    # Relevant + redundant features 'extended' to n_obs
    rr_exp = np.vstack([np.repeat(rr, q, axis=0),rr[:r,:]])

    # Creating irrelevant features
    irlvnt = np.random.randint(2, size=[n_obs,n_I])

    # Creating target values
    y = lor(
        land(rr_exp[:,0], rr_exp[:,1]),
        land(rr_exp[:,2], rr_exp[:,3])
    ).astype(int)

    # Making correlated features
    cor = make_cor(y)

    features = np.hstack([rr_exp, cor, irlvnt])

    return features, y

def make_cor_adv(y, n_class=4):
    n_ind = int(0.3*len(y))
    cor_vars = []
    for i in range(2):
        random.seed(0)
        np.random.seed(0)
        cor_i = y.copy()
        ind = random.sample(range(len(y)), n_ind)
        adjust = np.random.randint(n_class, size=n_ind)
        cor_i[ind] = (cor_i[ind]+adjust)%n_class
        cor_vars.append(cor_i)

    return np.array(cor_vars).transpose()

# ADDER dataset + Comparator
def adder(n_obs=50, n_I=92, seed=0):
    np.random.seed(seed)

    # Redundant features (negation of relevant features)
    red = lnot(gen_3()).astype(int)

    # Relevant + redundant features
    rr = np.hstack([gen_3(), red])

    # Quotient of n_obs divided by 8
    q = n_obs // 8

    # Remained of n_obs divided by 8
    r = n_obs % 8

    # Relevant + redundant features 'extended' to n_obs
    rr_exp = np.vstack([np.repeat(rr,q, axis=0), rr[:r,:]])

    # Creating irrelevant features
    irlvnt = np.random.randint(2, size=[n_obs,n_I])

    # Creating target values
    y1 = lxor(
      lxor(rr_exp[:,0], rr_exp[:,1]), rr_exp[:,2]
    ).astype(int)

    y2 = lor(
      land(rr_exp[:,0], rr_exp[:,1]), land(rr_exp[:,2], lxor(rr_exp[:,0], rr_exp[:,1]))
    ).astype(int)
    y = [y1[j] + 2*y2[j] for j in range(len(y1))]

    # Making correlated features
    cor = make_cor_adv(np.array(y))

    features = np.hstack([rr_exp, cor, irlvnt])

    return features, y

# Generate 10 random integers for different seeds to account for
# random variations
int_seeds = (np.random.default_rng(seed=0)).integers(0, 10000, 10)

contsynt_datasets = defaultdict()
discsynt_datasets = defaultdict()

for i, s in enumerate(int_seeds):
    # ANDOR dataset with binary features
    X_ANDOR, y_ANDOR = andor(n_obs=50, n_I=90, seed=s)
    X_contANDOR = reconstruct_continuousVariables(X_ANDOR, seed=s)
    
    # ADDER dataset with multiclass features
    X_ADDER, y_ADDER = adder(n_obs=50, n_I=92, seed=s)
    X_contADDER = reconstruct_continuousVariables(X_ADDER, seed=s)

    contsynt_datasets[i] = {"ANDOR": X_contANDOR, "ADDER": X_contADDER}
    discsynt_datasets[i] = {"ANDOR": X_ANDOR, "ADDER": X_ADDER}

with open("cont_synthetic_datasets.pkl", "wb") as handle:
    pickle.dump(contsynt_datasets, handle)
with open("disc_synthetic_datasets.pkl", "wb") as handle:
    pickle.dump(discsynt_datasets, handle)

sys.exit()
