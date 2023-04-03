import self as self
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

X = np.loadtxt("/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step5.csv", np.float32)
y = np.loadtxt("/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step6.csv",
               np.float32)
Path = '/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step5.csv'


# dataset definition
class SpiralData(Dataset):
    # load the dataset
    def __init__(self, X, y):
        # store the inputs and outputs
        self.X = X
        self.y = y

    # number of rows in the dataset
    def __len__(self):
        return len(self.X), len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


# # create the dataset
# dataset = SpiralData(X, y)
#
# print('data', dataset.X)
# # select rows from the dataset
# train, test = random_split(dataset, [0.7, 0.3])
# # create a data loader for train and test sets
# train_dl = DataLoader(train, batch_size=10, shuffle=True)
# test_dl = DataLoader(test, batch_size=10, shuffle=False)
#
# print('Test', test_dl)
# print('train', len(train_dl))


class SpiralModel(nn.Sequential):
    def __init__(self):
        super(SpiralModel, self).__init__()
        self.linear1 = nn.Linear(2, 24)
        self.r1 = nn.ReLU()
        self.linear2 = nn.Linear(24, 12)
        self.r2 = nn.ReLU()
        self.linear3 = nn.Linear(12, 6)
        self.r3 = nn.ReLU()
        self.linear4 = nn.Linear(6, 2)

    def forward(self, X):
        X = self.linear1(X)
        X = self.r1(X)
        X = self.linear2(X)
        X = self.r2(X)
        X = self.linear3(X)
        X = self.r3(X)
        X = self.linear4(X)
        return X


def prepare_data_x(X):
    dataset1 = X
    train, test = random_split(dataset1, [0.7, 0.3])
    train_dlx = DataLoader(train, batch_size=10, shuffle=True)
    test_dlx = DataLoader(test, batch_size=10, shuffle=False)

    return train_dlx, test_dlx


def prepare_data_y(y):
    dataset2 = y
    train, test = random_split(dataset2, [0.7, 0.3])
    train_dly = DataLoader(train, batch_size=10, shuffle=True)
    test_dly = DataLoader(test, batch_size=10, shuffle=False)

    return train_dly, test_dly


# train the model
optimizer = optim.Adam(model.parameters(), lr=0.0055)


def train_model(train_dl, model):
    # define the optimization
    loss = nn.MSEloss(ypred, train_dl)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(150):
        # enumerate mini batches
        for i, (train_dl, y) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            ypred = model(train_dl)
            # calculate loss
            loss = criterion(ypred, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
