import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_path_train = "/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step2.csv"
data = np.loadtxt(data_path_train)
print(data)
nx = []
ny = []


# nx = np.zeros(len(data))
# print(nx.shape)
# ny = np.zeros(len(data))
# print(ny.shape)

def min_max(data):
    x_min = data[:, 0].min()
    x_max = data[:, 0].max()
    y_min = data[:, 1].min()
    y_max = data[:, 1].max()
    return x_min, x_max, y_min, y_max


# print(min_max(data))
# print(data[0])


def prep_for_bins(nx, ny, data):
    x_min, x_max, y_min, y_max = min_max(data)
    dx = x_max - x_min
    dy = y_max - y_min
    for i, val in enumerate(data):
        nx.append(((val[0] - x_min) / dx))
        ny.append((val[1] - y_min) / dy)
    return nx, ny


nx, ny = prep_for_bins(nx, ny, data)


nx = np.dot(nx, 10)
ny = np.dot(ny, 10)


nx_ny= [nx,ny]
nx_ny_floor= np.floor(nx_ny)
print('this is', nx_ny_floor)


bins= np.zeros((10,10))
print(bins)

def average_vec(nx_ny, nx_ny_floor, bins):
    for i in nx_ny_floor:



