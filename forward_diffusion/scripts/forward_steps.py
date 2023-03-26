import numpy as np
import matplotlib.pyplot as plt
from generate_distribution import dist

def add_noise_and_print( mu, std,dist):
    noisex = np.random.normal(mu, std, len(dist))
    noisey = np.random.normal(mu, std, len(dist))
    noisey_dist_x = dist[:, 0] * np.sqrt(1-std) + noisex
    noisey_dist_y = dist[:, 1] * np.sqrt(1-std) + noisey
    noise_1 = np.column_stack((noisey_dist_x, noisey_dist_y))
    print('New', noise_1)

    plt.scatter(noise_1[:, 0], noise_1[:, 1], alpha=0.1)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.show()

    for i in range(2,20):
        noisex = np.random.normal(mu, std, len(dist))
        noisey = np.random.normal(mu, std, len(dist))
        # noisey_dist_x += noisex
        # noisey_dist_y += noisey
        noisey_dist_x = noisey_dist_x * np.sqrt(1 - std) + noisex
        noisey_dist_y = noisey_dist_y * np.sqrt(1 - std) + noisey
        noise_1 = np.column_stack((noisey_dist_x, noisey_dist_y))
        print('New', noise_1)

        plt.scatter(noise_1[:, 0], noise_1[:, 1], alpha=0.1)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.show()
        plt.savefig('../models3/data/step' +str(i) + '.png')
        plt.clf()
        np.savetxt('..//models3/data/step' + str(i) + '.csv', np.c_[noisey_dist_x,noisey_dist_y])



mu = 0.0
std = 0.05
add_noise_and_print(mu, std, dist)

print('old', dist)
