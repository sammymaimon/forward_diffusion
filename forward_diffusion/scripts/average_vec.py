import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_path_train1 = "/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step2.csv"
data = np.loadtxt(data_path_train1)
data_path_train2 = "/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step3.csv"
data2 = np.loadtxt(data_path_train2)
print(data)
nx = []
ny = []

vec = (data - data2)
print("this is vec", vec)

# arrow = np.zeros(vec.shape[0])
#
# for i in range(vec.shape[0]):
#     arrow[i] = np.sqrt((vec[i, 0] ** 2 + vec[i, 1] ** 2))
#
# print('this is arrow', arrow)
#
# data = np.array([(data[i, 0], data[i, 1], vec[i, 0], vec[i, 1]) for i in range(len(data))])
# print(type(data))
# print(data.shape)


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
nx_ny = np.array([nx, ny]).T
nx_ny_floor = np.floor(nx_ny)
print(type(nx_ny_floor))

floor_and_vec = np.array([(nx_ny_floor[i, 0], nx_ny_floor[i, 1], vec[i, 0], vec[i, 1]) for i in range(len(nx_ny_floor))])


bins_x = np.zeros((10, 10))
bins_y = np.zeros((10, 10))

print(type(bins_x))

for i in range(len(floor_and_vec[0])):
    x = floor_and_vec[i, 0]
    y = floor_and_vec[i, 1]
    bins_x[x, y] += floor_and_vec[i, 2]
    bins_y[x, y] += floor_and_vec[i, 3]

print(bins_x)
