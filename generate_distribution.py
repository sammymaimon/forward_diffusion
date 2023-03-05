import matplotlib.pyplot as plt
from my_metropolis import metropolis
from sammys_spiral import spiral

dist = metropolis(spiral,0.0,0.0,0.1,10000,10)
plt.scatter(dist[:,0], dist[:,1], alpha=0.1)
plt.show()



#np.savetxt('n' + str(n) + '_j' + str(j) + '_runs' + str(n_runs) + '.dat', np.c_[, average_spectrum])