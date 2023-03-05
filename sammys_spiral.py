import numpy as np


def spiral(x, y, w=0.1, a=0.1, b=0.5):
    Theta = np.arctan2(x, y)
    R = np.sqrt(x**2 + y**2)

    Rs0 = a * Theta
    Rs1 = a * (Theta + 2*np.pi)
    Rs2 = a * (Theta + 4*np.pi)

    F = np.exp(-((R-Rs0)/w)**2) + np.exp(-((R-Rs1)/w)**2) + np.exp(-((R-Rs2)/w)**2)
    F = F * np.exp(-(R / b)** 2)
    return F


