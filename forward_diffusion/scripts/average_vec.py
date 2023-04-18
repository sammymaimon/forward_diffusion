import numpy as np
import matplotlib.pyplot as plt

data_path_train1 = "/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step2.csv"
data = np.loadtxt(data_path_train1)
data_path_train2 = "/home/sammy/PycharmProjects/pythonProject/forward_diffusion/models3/data/step3.csv"
data2 = np.loadtxt(data_path_train2)
nx = []
ny = []
EPSILON = 10e-5
bins_x = np.zeros((10, 10))
bins_y = np.zeros((10, 10))
bins_occ = np.zeros((10, 10))
vec = (data - data2)


def min_max(data):
    x_min = data[:, 0].min()
    x_max = data[:, 0].max()
    y_min = data[:, 1].min()
    y_max = data[:, 1].max()
    return x_min, x_max, y_min, y_max


def prep_for_bins(nx, ny, EPSILON, data):
    x_min, x_max, y_min, y_max = min_max(data)
    dx = x_max - x_min + EPSILON
    dy = y_max - y_min + EPSILON
    for i, val in enumerate(data):
        nx.append(((val[0] - x_min) / dx))
        ny.append((val[1] - y_min) / dy)
    return nx, ny


nx, ny = prep_for_bins(nx, ny, EPSILON, data)
nx = np.dot(nx, 10)
ny = np.dot(ny, 10)
nx_ny = np.array([nx, ny]).T
nx_ny_floor = np.floor(nx_ny)

floor_and_vec = np.array(
    [(nx_ny_floor[i, 0], nx_ny_floor[i, 1], vec[i, 0], vec[i, 1]) for i in range(len(nx_ny_floor))]).T

for i in range(len(floor_and_vec[0])):
    x = int(floor_and_vec[0, i])
    y = int(floor_and_vec[1, i])
    bins_x[x, y] += floor_and_vec[2, i]
    bins_y[x, y] += floor_and_vec[3, i]
    bins_occ[x, y] += 1

bins_occ = bins_occ + EPSILON
average_movement = bins_x / bins_occ

print(average_movement)

plt.figure()
plt.imshow(bins_x)
plt.figure()
plt.imshow(average_movement)

plt.show()
