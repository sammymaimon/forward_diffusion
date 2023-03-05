import numpy as np

def metropolis(p, x0, y0, r0, n, decorr=10):
  x, y = x0, y0
  n_samples = 0
  distribution = np.zeros([n, 2])
  step_num = 0
  while n_samples < n:
    xp = x + r0 * (0.5 - np.random.rand(1)[0])*2
    yp = y + r0 * (0.5 - np.random.rand(1)[0])*2
    acceptance_probability = p(xp, yp) / p(x, y)
    if acceptance_probability >= np.random.rand(1):
      x = xp
      y = yp
    if step_num % decorr == 0:
      distribution[n_samples] = xp, yp
      n_samples += 1
    step_num += 1
  return distribution

