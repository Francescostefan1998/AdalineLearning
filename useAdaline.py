import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from adalineSqError import AdalineGD 
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

fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize=(16, 4))
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X,y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')


ada2 = AdalineGD(n_iter=15, eta=0.01).fit(X,y)
ax[1].plot(range(1, len(ada2.losses_) + 1), np.log10(ada2.losses_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title('Adaline - Learning rate 0.01')

ada3 = AdalineGD(n_iter=15, eta=0.001).fit(X,y)
ax[2].plot(range(1, len(ada3.losses_) + 1), np.log10(ada3.losses_), marker='o')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('log(Mean squared error)')
ax[2].set_title('Adaline - Learning rate 0.001')

ada4 = AdalineGD(n_iter=15, eta=0.0001).fit(X,y)
ax[3].plot(range(1, len(ada4.losses_) + 1), np.log10(ada4.losses_), marker='o')
ax[3].set_xlabel('Epochs')
ax[3].set_ylabel('log(Mean squared error)')
ax[3].set_title('Adaline - Learning rate 0.0001')
 
ada5 = AdalineGD(n_iter=95, eta=0.0108).fit(X,y)
ax[4].plot(range(1, len(ada5.losses_) + 1), np.log10(ada5.losses_), marker='o')
ax[4].set_xlabel('Epochs')
ax[4].set_ylabel('log(Mean squared error)')
ax[4].set_title('Adaline - Learning rate 0.0108')
# Adjust layout for better spacing
plt.tight_layout()

plt.show()


#this use the standardization to center better for each weight the learning rate
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:,0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:,1].mean())/X[:, 1].std()

ada_gd = AdalineGD(n_iter = 20, eta=0.5)
ada_gd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_gd.losses_) + 1),
         ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
plt.show()






