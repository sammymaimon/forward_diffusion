import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split


X = np.loadtxt("/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step5.csv")
y = np.loadtxt("/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step6.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# set up DataLoader for training set
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=10000)

print(loader)

n_epochs = 20
history = []
y_pred_save = []




# X_train = X_train.clone().detach()
# y_train = y_train.clone().detach()
# X_test = X_test.clone().detach()
# y_test = y_test.clone().detach()

model = nn.Sequential(
    nn.Linear(2, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 2)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0055)
print(len(X_test))
for epoch in range(n_epochs):
    for x_batch, y_batch in loader:
        print(x_batch.shape)
        model.train()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        y_pred_save = y_pred

y_numpy = y_pred_save.detach().numpy()
x_numpy = X_test.detach().numpy()

plt.plot(history)
plt.show()

# plt.scatter(y_numpy[:,0],y_numpy[:,1])
# plt.show()

plt.quiver(x_numpy[:, 0], x_numpy[:, 1], y_numpy[:, 0], y_numpy[:, 1])
plt.plot(x_numpy[:, 0], x_numpy[:, 1], '.')
plt.show()
