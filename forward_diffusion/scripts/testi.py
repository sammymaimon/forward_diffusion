import matplotlib.pyplot as plt
import numpy as np
import glob
import os

fig = plt.figure()
ax = fig.add_subplot(111)
#title = 'c0={c0}, c1={c1}, duration={t_run}'.format(c0=complex(round(c0.real, 5), round(c0.imag, 5)), c1=complex(round(c1.real, 5), round(c1.imag, 5)), t_run=str(t_max))
# title = 'Ψ_exact - Ψ_numerical'
#fig.suptitle(title)
ax.set_ylim([-1.5, 1.5])
ax.set_xlim([-1.5, 1.5])
ax.grid(True, which='both', ls='--')
ax.axhline(y=0, color='k', alpha=0.75)
ax.set_axisbelow(True)
ax.set_xlabel("X")


file_list = glob.glob(os.path.join(os.getcwd(), "*.txt"))

corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())

corpus_new = []
add_i = []
for i in corpus:
    add_j = []
    for j in i.split('\n'):
        print('jsplit', j.split('\n'))
        add_j.append([float(x) for x in j.split(' ')])
    add_i.append(add_j)
    corpus_new.append(add_i[:-1])
corpus_new = np.array(corpus_new[1:-1])
print(corpus_new)

# y_err = exact_phi(x, time_list[0], c0=c0, c1=c1, k=k, m=m) - y
line1, = ax.scatter()
ax.legend()

def animate(i):
    line1.set_ydata(step[i][0])
    line1.set_xdata(step[i][1])

    return line1

ani = FuncAnimation(fig, animate, frames=20, blit=True)
ani.save("name.gif", dpi=250, writer=PillowWriter(fps=50))
