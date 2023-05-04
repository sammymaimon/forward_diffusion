import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split


N = 1000
mu = 0
std = 2e-1
noise = []


# X1_plot, X2_plot = np.meshgrid(np.linspace(-1., 1., N), np.linspace(-1., 1., N))
# Z = torch.tensor(np.stack([X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2)]), dtype= torch.float32).T
noise = np.array((np.random.normal(mu, std, N ** 2), np.random.normal(mu, std, N ** 2))).T
noise_shifted_rand = noise.copy()
noise_shifted = noise.copy()

# plt.scatter(noise[:,0],noise[:,1])
# plt.show()

for i in range(len(noise)):
    if noise[i, 0] <= 0:
        noise_shifted_rand[i, 0] -= 0.1 + np.random.normal(0, 0.01)
    elif noise[i, 0] > 0:
        noise_shifted_rand[i, 0] += 0.1 + np.random.normal(0, 0.01)

# for i in range(len(X)):
#     if X[i, 0] <= 0:
#         X_shifted[i, 0] -= 0.1
#     elif X[i, 0] > 0:
#         X_shifted[i, 0] += 0.1

# plt.scatter(noise[:,0],noise[:,1])
# plt.scatter(noise_shifted[:,0],noise_shifted[:,1])
# plt.show()


X, y = noise_shifted_rand, noise
n_epochs = 150
history = []
y_pred_save = []


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


X_train = X_train.clone().detach()
y_train = y_train.clone().detach()
X_test = X_test.clone().detach()
y_test = y_test.clone().detach()

loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=10000)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    # nn.Linear(16, 12),
    # nn.ReLU(),
    # nn.Linear(12, 6),
    # nn.ReLU(),
    nn.Linear(16, 2)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0055)

for epoch in range(n_epochs):
    for x_batch, y_batch in loader:
        model.train()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

        model.eval()
        y_pred = model(y_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        y_pred_save = y_pred

# torch.save(model, '../tests/model_tests/model_for_randomtest3.pt')



y_numpy = y_pred_save.detach().numpy()
x_numpy = X_test.detach().numpy()

plt.plot(history)
plt.show()


def arrow_plot(ax: plt.Axes, X, y, n=None, color='k'):
    if n is None:
        n = X.shape[0]
    for i in range(n):
        dx = y[i, 0] - X[i, 0]
        dy = y[i, 1] - X[i, 1]
        ax.arrow(X[i, 0], X[i, 1], dx, dy, color=color, alpha=0.25)


n_show = 400
fig = plt.figure()
ax = plt.gca()
ax.plot(x_numpy[:n_show, 0], x_numpy[:n_show, 1], '.', color='k')
arrow_plot(ax, x_numpy, y_numpy, n=n_show, color='r') # predicted values
arrow_plot(ax, x_numpy, y_test.detach().numpy(), n=n_show, color='k') # expected values
plt.show()





Q1 = sum(abs((y_numpy - y_test.detach().numpy())))/len(y_numpy)
print("this is q1", Q1)