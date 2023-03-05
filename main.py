import numpy as np
import matplotlib.pyplot as plt


def normal(x, mu, sigma):
    numerator = np.exp((-(x - mu) ** 2) / (2 * sigma ** 2))
    denominator = sigma * np.sqrt(2 * np.pi)
    return numerator / denominator


def random_coin(p):
    unif = random.uniform(0, 1)
    if unif >= p:
        return False
    else:
        return True


def gaussian_mcmc(hops, mu, sigma):
    states = []
    burn_in = int(hops * 0.2)
    current = random.uniform(-5 * sigma + mu, 5 * sigma + mu)
    for i in range(hops):
        states.append(current)
        movement = random.uniform(-5 * sigma + mu, 5 * sigma + mu)

        curr_prob = normal(x=current, mu=mu, sigma=sigma)
        move_prob = normal(x=movement, mu=mu, sigma=sigma)

        acceptance = min(move_prob / curr_prob, 1)
        if random_coin(acceptance):
            current = movement
    return states[burn_in:]


lines = np.linspace(-3, 3, 1000)
normal_curve = [normal(l, mu=0, sigma=1) for l in lines]
dist = gaussian_mcmc(100_000, mu=0, sigma=1)
plt.hist(dist, normed=1, bins=20)
plt.plot(lines, normal_curve)