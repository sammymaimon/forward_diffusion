import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def f(x):
   return 0.274*np.exp(-0.86*x**2)+0.117*np.exp(-0.15*x**2)

def g(x):
    return 0.059*np.exp(-0.11*x**2)+0.194*np.exp(-0.41*x**2) +0.195*np.exp(-2.23*x**2)

def s(x):
    return (np.sqrt(1/np.pi))*np.exp(-x)

def h(x):
    return 0.267*np.exp(-0.27*x**2)

x = np.linspace(0, 5, 1000)

plt.plot(x, f(x), color='red')
plt.plot(x, s(x), color='blue')
plt.plot(x, g(x), color='black')
plt.plot(x, h(x), color='green')
plt.show()