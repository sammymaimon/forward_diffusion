import numpy as np
import matplotlib.pyplot as plt
from generate_distribution import dist

def add_noise_and_print(mu, std, dist):
    noisex = np.random.normal(mu, std, len(dist))
    noisey = np.random.normal(mu, std, len(dist))
    noisey_dist_x = dist[:, 0] + noisex
    noisey_dist_y = dist[:, 1] + noisey
    noise_1 = np.column_stack((noisey_dist_x, noisey_dist_y))
    print('New', noise_1)

    plt.scatter(noise_1[:, 0], noise_1[:, 1], alpha=0.1)
    plt.show()

    for i in range(2,6):
        noisey_dist_x += noisex
        noisey_dist_y += noisey
        noise_1 = np.column_stack((noisey_dist_x, noisey_dist_y))
        print('New', noise_1)

        plt.scatter(noise_1[:, 0], noise_1[:, 1], alpha=0.1)
        plt.show()



mu = 0.02
std = 5e-2
add_noise_and_print(mu, std, dist)

print('old', dist)




