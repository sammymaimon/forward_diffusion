import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

for i in range(19, 2, -1):
    # for i in [6]:
    # T = ('../models3/data/step' + str(i) + '.csv')
    # t_1 = '../models3/data/step' + str(i - 1) + '.csv'
    #
    # spiral_data_numpy = np.loadtxt(T, np.float32)
    # spiral_answer_numpy = np.loadtxt(t_1, np.float32)
    #
    # Train_data = torch.from_numpy(spiral_data_numpy)
    # Answer_data = torch.from_numpy(spiral_answer_numpy)
    #
    # X, y = Train_data, Answer_data

    X = np.loadtxt('../models3/data/step' + str(i) + '.csv', dtype=np.float32)
    y = np.loadtxt('../models3/data/step' + str(i - 1) + '.csv', dtype=np.float32)

    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    # set up DataLoader for training set
    loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=10000)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    n_epochs = 150
    history = []
    # y_pred = []

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

    for epoch in range(n_epochs):
        for x_batch, y_batch in loader:
            model.train()
            print(x_batch.shape)
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

    torch.save(model, '../models4/' + '_' + str(i)+'_' + str(i-1) + '.pt')

    # N = 40
    # X1_plot, X2_plot = np.meshgrid(np.linspace(-1., 1., N), np.linspace(-1., 1., N))
    # X_plot = torch.tensor(np.stack([X1_plot.reshape(N ** 2), X2_plot.reshape(N ** 2)]), dtype=torch.float32).T
    # Y_plot = model(X_plot).detach().numpy()
    # X_plot = X_plot.detach().numpy()
    # y_numpy = y_pred_save.detach().numpy()
    # print(type(y_numpy))
    # x_numpy = X_test.detach().numpy()
    # plt.quiver(X_plot[:, 0], X_plot[:, 1], Y_plot[:, 0], Y_plot[:, 1])
    # plt.plot(X_plot[:, 0], X_plot[:, 1], '.')

    # plt.show()

    Y_plot = model(X_test).detach().numpy()
    plt.figure()
    plt.quiver(X_test[:, 0], X_test[:, 1], y_test[:, 0], y_test[:, 1])
    plt.quiver(X_test[:, 0], X_test[:, 1], Y_plot[:, 0], Y_plot[:, 1], color='red', alpha=0.5)
    plt.plot(X_test[:, 0], X_test[:, 1], '.', alpha= 0.5)
    plt.savefig('../models4/' + '_' + str(i)+'_' + str(i-1) + '.png')
    # plt.plot(y_test[:, 0], y_test[:, 1], '.')

plt.show()

# plt.plot(history)
# plt.show()

# plt.scatter(y_numpy[:, 0], y_numpy[:, 1])
# plt.show()

# plt.quiver(x_numpy[:, 0], x_numpy[:, 1], y_numpy[:, 0], y_numpy[:, 1])
# plt.plot(x_numpy[:, 0], x_numpy[:, 1], '.')
# plt.show()
