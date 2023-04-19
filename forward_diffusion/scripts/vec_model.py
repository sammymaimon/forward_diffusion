from average_vec import average_movement_x
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader


N=10


X1_plot, X2_plot = np.meshgrid(np.linspace(-1., 1., N), np.linspace(-1., 1., N))
X = torch.tensor(np.stack([X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2)]), dtype=torch.float32).T
plt.scatter(X[:,0], X[:, 1])
plt.show()
Y = X.clone().detach()

model = torch.load("../models4/_3_2.pt")
model.eval()

Y = model(X)
Y_plot = (Y.clone().detach().numpy())

plt.figure()
plt.scatter(Y_plot[:, 0], Y_plot[:, 1], s=None)

def arrow_plot(ax: plt.Axes, X, y, n=None, color='k'):
    if n is None:
        n = X.shape[0]
    for i in range(n):
        dx = y[i, 0] - X[i, 0]
        dy = y[i, 1] - X[i, 1]
        ax.arrow(X[i, 0], X[i, 1], dx, dy, color=color)


n_show = 1000
fig = plt.figure()
ax = plt.gca()
ax.plot(x_numpy[:n_show, 0], x_numpy[:n_show, 1], '.', color='k')
# ax.plot(y_test[:n_show, 0], y_test[:n_show, 1], '.', color = 'b')
ax.plot(y_train[:n_show, 0], y_train[:n_show, 1], '.', color='b')
arrow_plot(ax, x_numpy, y_numpy, n=n_show, color='r')
# arrow_plot(ax, x_numpy, y_test, n=n_show, color='k')
arrow_plot(ax, x_numpy, y_train, n=n_show, color='k')
plt.show()