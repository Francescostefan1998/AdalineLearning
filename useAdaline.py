import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from adalineGD import AdalineGD 
from matplotlib.colors import ListedColormap



# Hardcoded dataset
X = np.array([
    [5, 10],  # Example 1: 5 hours screen time, 10k steps = REGULAR 1
    [7, 5],   # Example 2: 7 hours screen time, 5k steps = NOT REGULAR 0
    [3, 12],  # Example 3: 3 hours screen time, 12k steps = REGULAR 1
    [8, 3],   # Example 4: 8 hours screen time, 3k steps = NOT REGULAR 0
    [6, 8],   # Example 5: 6 hours screen time, 8k steps = REGULAR 1
    [4, 7],   # Example 6: 4 hours screen time, 7k steps = REGULAR 1
    [9, 4],   # Example 7: 9 hours screen time, 4k steps = NOT REGULAR 0
    [2, 11],  # Example 8: 2 hours screen time, 11k steps = REGULAR 1
    [6, 6],   # Example 9: 6 hours screen time, 6k steps = NOT REGULAR 0
    [5, 9]    # Example 10: 5 hours screen time, 9k steps = REGULAR 1
])

# Labels (y)
y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])  # 1 = Regular, 0 = Not Regular

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=3, eta=0.1).fit(X,y)
wBef1 = ada1.w_
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ada2 = AdalineGD(n_iter=3, eta=0.0001).fit(X, y)
wBef2 = ada2.w_

ax[1].plot(range(1, len(ada2.losses_) +1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
print(f' Weights f 1 before adg {wBef1}')
print(f' Weights f 1 after adg {ada1.w_}')
print(f' Weights f 2 before adg {wBef2}')
print(f' Weights f 2 after adg {ada1.w_}')



