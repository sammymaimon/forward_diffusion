import numpy as np
import matplotlib.pyplot as plt
import random
#def spiral(t, a, b):
#  x = a * np.cos(t) * np.exp(b * t)
#  y = a * np.sin(t) * np.exp(b * t)
#  return x, y


#t = np.linspace(0, 10, 100)
#x, y = spiral(t, a=1, b=0.1)
#plt.plot(x, y)
#plt.show()



import random
import numpy as np
import matplotlib.pyplot as plt

def spiral(t, a, b):
  x = a * np.cos(t) * np.exp(b * t)
  y = a * np.sin(t) * np.exp(b * t)
  return x, y


t = np.linspace(0, 10, 100)
x, y = spiral(t, a=1, b=0.1)
plt.plot(x, y)
plt.show()

def target_distribution(sample):
  # Calculate the probability of the sample under the target distribution
  # You can use the spiral function to calculate the probability of the sample
  x, y = sample
  t = np.arctan2(y, x)
  a = 1
  b = 0.1
  x_, y_ = spiral(t, a, b)
  probability = np.exp(-0.5 * ((x - x_) ** 2 + (y - y_) ** 2))
  return probability

def proposal_distribution(current_sample):
  # Sample from a Gaussian distribution centered around the current sample
  x, y = current_sample
  x_proposed = random.gauss(x, 0.1)
  y_proposed = random.gauss(y, 0.1)
  return x_proposed, y_proposed

def metropolis_hastings(target_distribution, proposal_distribution, num_samples):
  # Initialize the list of samples
  samples = []

  # Choose an initial sample from the proposal distribution
  current_sample = proposal_distribution((0, 0))

  for i in range(num_samples):
    # Propose a new sample from the proposal distribution
    proposed_sample = proposal_distribution(current_sample)

    # Calculate the acceptance probability of the proposed sample
    acceptance_probability = min(1, target_distribution(proposed_sample) / target_distribution(current_sample))

    # Generate a random number between 0 and 1
    u = random.uniform(0, 1)

    # If the random number is less than the acceptance probability, accept the proposed sample
    if u < acceptance_probability:
      current_sample = proposed_sample

    # Add the current sample to the list of samples
    samples.append(current_sample)

  return samples

# Generate 1000 samples from the spiral function
samples = metropolis_hastings(target_distribution, proposal_distribution, 1000000)

# Plot a histogram of the samples
x, y = zip(*samples)
plt.hist(x, bins=100)
plt.show()

