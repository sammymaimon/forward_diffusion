import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

N = 10

X1_plot, X2_plot = np.meshgrid(np.linspace(-1., 1., N), np.linspace(-1., 1., N))
X = (np.stack([X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2)])).T
X_shifted_rand = X.copy()
X_shifted = X.copy()
history = []
loss_fn = nn.MSELoss()

for i in range(len(X)):
    if X[i, 0] <= 0:
        X_shifted_rand[i, 0] -= 0.1 + np.random.normal(0, 0.01)
    elif X[i, 0] > 0:
        X_shifted_rand[i, 0] += 0.1 + np.random.normal(0, 0.01)

for i in range(len(X)):
    if X[i, 0] <= 0:
        X_shifted[i, 0] -= 0.1
    elif X[i, 0] > 0:
        X_shifted[i, 0] += 0.1



# X = torch.tensor(np.stack([X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2)]), dtype=torch.float32).T
X_shifted_rand = torch.tensor(X_shifted_rand, dtype=torch.float32)
model = torch.load("../tests/model_tests/model_for_randomtest1.pt")
model.eval()

y_pred = model(X_shifted_rand)

# mse = loss_fn(y_pred, X_shifted)
# mse = float(mse)
# history.append(mse)
# plt.semilogy(history)
# plt.show()


y_numpy = y_pred.detach().numpy()
x_numpy = X_shifted_rand.detach().numpy()



def arrow_plot(ax: plt.Axes, X, y, n=None, color='k'):
    if n is None:
        n = X.shape[0]
    for i in range(n):
        dx = y[i, 0] - X[i, 0]
        dy = y[i, 1] - X[i, 1]
        ax.arrow(X[i, 0], X[i, 1], dx, dy, color=color, alpha=0.25)


n_show = N ** 2
fig = plt.figure()
ax = plt.gca()
ax.plot(X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2), '.', color='k')
arrow_plot(ax, X, y_numpy, n=n_show, color='r')  # predicted values
arrow_plot(ax, X, X_shifted, n=n_show, color='k')  # expected values
plt.show()

Q1 = abs(sum((y_numpy - X_shifted)))/len(y_numpy)
print("this is q1", Q1)