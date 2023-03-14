import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_path_train = "../images/txt/step3.csv"
data_path_answer = "../images/txt/step2.csv"

spiral_data_numpy = np.loadtxt(data_path_train, np.float32)
spiral_answer_numpy = np.loadtxt(data_path_answer, np.float32)

Train_data = torch.from_numpy(spiral_data_numpy)
Answer_data = torch.from_numpy(spiral_answer_numpy)

X, y = Train_data, Answer_data
n_epochs = 125
history = []
y_pred_save = []

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


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
optimizer = optim.Adam(model.parameters(), lr = 0.0055)



for epoch in range(n_epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
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


# plt.plot(history)
# plt.show()

# plt.scatter(y_numpy[:,0],y_numpy[:,1])
# plt.show()

plt.quiver(x_numpy[:,0], x_numpy[:,1], y_numpy[:,0], y_numpy[:,1])
#plt.plot(x_numpy[:,0], x_numpy[:,1], '.')
plt.show()