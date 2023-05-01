import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from froward_steps import add_noise_and_print
from generate_distribution import dist

mu = 0.2
std= 0.01

create_data=[]
create_data.append([add_noise_and_print(dist, mu, std)])
print('did this work', create_data)
