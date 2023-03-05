import numpy as np
from matplotlib import pyplot as plt

w=0.1
a=0.1
b = 0.5

x = np.linspace(-1.0, 1.0, 1000)
X, Y = np.meshgrid(x, x)
Theta = np.arctan2(X, Y)
R = np.sqrt(X**2 + Y**2)

Rs0 = a * Theta
Rs1 = a * (Theta + 2*np.pi)
Rs2 = a * (Theta + 4*np.pi)

F = np.exp(-((R-Rs0)/w)**2) + np.exp(-((R-Rs1)/w)**2) + np.exp(-((R-Rs2)/w)**2)
F = F * np.exp(-(R / b)** 2)

plt.contourf(X, Y, F, levels=100)
plt.colorbar()
plt.show()

