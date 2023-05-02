import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

N=10

X1_plot, X2_plot = np.meshgrid(np.linspace(-1., 1., N), np.linspace(-1., 1., N))
X = (np.stack([X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2)])).T
X_shifted = X.copy()


for i in range(len(X)):
    if X[i, 0] <= 0:
        X_shifted[i, 0] -= 0.1 + np.random.normal(0, 0.02)
    elif X[i, 0] > 0:
        X_shifted[i, 0] += 0.1 + np.random.normal(0, 0.02)


print(X-X_shifted)
# exit()

# data_path_train1 = "/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step2.csv"
# data = np.loadtxt(data_path_train1)
# data_path_train2 = "/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step3.csv"
# data2 = np.loadtxt(data_path_train2)
X = torch.tensor(np.stack([X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2)]), dtype= torch.float32).T
model = torch.load("../tests/model_tests/model_for_randomtest1.pt")
model.eval()

y_pred = model(X)

y_numpy = y_pred.detach().numpy()
x_numpy = X.detach().numpy()

def arrow_plot(ax: plt.Axes, X, y, n=None, color='k'):
    if n is None:
        n = X.shape[0]
    for i in range(n):
        dx = y[i, 0] - X[i, 0]
        dy = y[i, 1] - X[i, 1]
        ax.arrow(X[i, 0], X[i, 1], dx, dy, color=color, alpha=0.25)


n_show = N**2
fig = plt.figure()
ax = plt.gca()
ax.plot(X1_plot.reshape(N**2), X2_plot.reshape(N**2), '.', color='k')
arrow_plot(ax, x_numpy, y_numpy, n=n_show, color='r') # predicted values
arrow_plot(ax, x_numpy, X_shifted, n=n_show, color='k') # expected values
plt.show()


print(x_numpy-y_numpy)