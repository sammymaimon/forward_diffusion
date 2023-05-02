import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

N = 40
mu = 0.02
std = 2e-1
noise = []

noise = np.array((np.random.normal(mu, std, N ** 2), np.random.normal(mu, std, N ** 2))).T

noise1 = noise + 0.1

# plt.scatter(noise[:,0], noise[:,1])
# plt.show()


X, y = noise, noise1
n_epochs = 1500
history = []
y_pred_save = []
val = []
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

#
# X_train = X_train.clone().detach()
# y_train = y_train.clone().detach()
# X_test = X_test.clone().detach()

# y_test = y_test.clone().detach()

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

loader = DataLoader(list(zip(X_train, y_train)), batch_size=1600)

model = nn.Sequential(
    nn.Linear(2, 24),
    nn.ReLU(),
    # nn.Linear(24, 12),
    # nn.ReLU(),
    # nn.Linear(12, 6),
    # nn.ReLU(),
    nn.Linear(24, 2)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0055)
val_loss = nn.MSELoss()

for epoch in range(n_epochs):
    for x_batch, y_batch in loader:
        model.train()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        loss = float(loss)
        val.append(loss)

        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        y_pred_save = y_pred

y_numpy = model(x_batch).detach().numpy()
x_numpy = x_batch.detach().numpy()

plt.semilogy(val)
plt.semilogy(history)
plt.show()


def arrow_plot(ax: plt.Axes, X, y, n=None, color='k'):
    if n is None:
        n = X.shape[0]
    for i in range(n):
        dx = y[i, 0] - X[i, 0]
        dy = y[i, 1] - X[i, 1]
        ax.arrow(X[i, 0], X[i, 1], dx, dy, color=color, alpha=0.5)


n_show = 1120
fig = plt.figure()
ax = plt.gca()
ax.plot(X[:n_show, 0], X[:n_show, 1], '.', color='k')
# ax.plot(y_test[:n_show, 0], y_test[:n_show, 1], '.', color = 'b')
# ax.plot(y_train[:n_show, 0], y_train[:n_show, 1], '.', color='b')
arrow_plot(ax, x_numpy, y_train, n=n_show, color='k')
arrow_plot(ax, x_numpy, y_numpy, n=n_show, color='r')
# # arrow_plot(ax, x_numpy, y_test, n=n_show, color='k')

plt.show()
