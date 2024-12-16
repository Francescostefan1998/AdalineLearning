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
# y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])  # 1 = Regular, 0 = Not Regular


s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print ('From URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
# print(df.tail())

#select setosa and versicolor
y = df.iloc[0:100, 4].values # so basically here I am extracting the labels which might be Iris-setos or iris-versicolor

y = np.where(y == 'Iris-setosa', 0, 1) # here is a condition if it is iris-setosa it will give a 0 otherwis 1 but I will get an array

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values # here is extracting just two properties from each row the first and the third [0, 2]
print('--------------------------------------------------------------------')
print(X)
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


def plot_decision_regions(X, y, classifier, resolution = 0.02):
    # setup marker generator and color map 
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[len(np.unique(y))])

    #plot the decision surpface
    x1_min, x1_max = X[:, 0].min() -1, X[:,0].max() +1
    x2_min, x2_max = X[:, 1].min() -1, X[:,1].max() +1

    xx1, xx2=np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                         np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha = 0.8,
                    c= colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
        
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






