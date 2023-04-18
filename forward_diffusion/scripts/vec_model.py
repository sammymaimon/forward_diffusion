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
print('this is Y_plot', Y_plot)
plt.figure()
plt.scatter(Y_plot[:, 0], Y_plot[:, 1], s=None)
plt.show()