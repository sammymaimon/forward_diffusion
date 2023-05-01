import numpy as np
import matplotlib.pyplot as plt
from generate_distribution import dist

mu = 0.02
std = 2e-1

print('old', dist)
def add_noise_x(mu, std, dist):
    noise = np.random.normal(mu, std, len(dist))
    noisey_dist_x = dist[:,0] + noise
    return noisey_dist_x


def add_noise_y(mu, std, dist):
    noise = np.random.normal(mu, std, len(dist))
    noisey_dist_y = dist[:,1] + noise
    return noisey_dist_y


noise_1 = np.column_stack((add_noise_x(mu,std,dist), add_noise_y(mu,std,dist)))

print('New', noise_1)

plt.scatter(noise_1[:,0], noise_1[:,1], alpha=0.1)
plt.show()
