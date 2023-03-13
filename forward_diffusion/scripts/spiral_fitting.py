import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_path_train ="../images/txt/step2.csv"
data_path_answer = "../images/txt/step3.csv"

spiral_data_numpy = np.loadtxt(data_path_train, np.float32)
spiral_answer_numpy = np.loadtxt(data_path_answer, np.float32)

Train_data = torch.from_numpy(spiral_data_numpy)
Answer_data = torch.from_numpy(spiral_answer_numpy)
X, y = Train_data, Answer_data
print((Train_data[0]), Answer_data[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print('test -', X_test)

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
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 100
history=[]


for epoch in range(n_epochs):
    model.train()
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
