import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Class separation
# Dataset parameters
separation = [0, 1, 2]

# Negative class
mu0 = 0
variance = 1
sigma = np.sqrt(variance)
X = np.linspace(mu0 - 3*sigma, 3 + 3*sigma, 100)

negativeClass = stats.norm.pdf(X, 0, 1.0)

for i, s in enumerate(separation):
    fig, ax = plt.subplots(1, 1)
    ax.plot(X, negativeClass, linewidth=3.0)

    positiveClass = stats.norm.pdf(X, s, 1.0)
    ax.plot(
        X, positiveClass, color="orange", linestyle="-."
    )
    plt.xticks(np.arange(-4, 7, 1), fontsize='x-large')
    plt.xlim((-3.5, 6.5))
    plt.yticks([])
    ax.grid(visible=True, which='major', axis='x')
    plt.tight_layout()
    fig.savefig(f"class_sep{s}.png", format="png")

# 2. Class imbalance
imbalances = [40, 45, 50]

for i, imba in enumerate(imbalances):
    fig, ax = plt.subplots(1, 1)
    negativeClass = stats.norm.pdf(X, 0, 1.0)*(1-imba/100)
    ax.plot(X, negativeClass, linewidth=3.0)

    positiveClass = stats.norm.pdf(X, 2.0, 1.0)*(imba/100)
    ax.plot(
        X, positiveClass, color="orange", linestyle="-."
    )
    plt.xticks(np.arange(-4, 7, 1), fontsize='x-large')
    plt.xlim((-3.5, 6.5))
    plt.yticks([])
    ax.grid(visible=True, which='major', axis='x')
    plt.tight_layout()
    fig.savefig(f"class_imbalance{imba}.png", format="png")

sys.exit()
