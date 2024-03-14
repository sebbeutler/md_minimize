import numpy as np
import math
import matplotlib.pyplot as plt

# Constants
L0 = 0.9572
KB = 450.0

D = 0.001
DT = 0.001

TEMPINIT = 5

# Function to calculate the total energy
def energie(p):
    return KB*(math.sqrt( p[0]**2 + p[1]**2) - L0)**2

# Function to calculate the gradient
def gradient(p):
    g = np.zeros_like(p)
    g[0] = -(energie(p + [D, 0]) - energie(p - [D, 0])) / (2*D)
    g[1] = -(energie(p + [0, D]) - energie(p - [0, D])) / (2*D)
    return g

if __name__ == '__main__':
    # Initial positions
    pos_prev = np.array([2.0, 2.0])
    gradient_init = gradient(pos_prev)

    pos = pos_prev + (TEMPINIT**.5 * DT) + (0.5 * gradient_init * DT**2)

    Epot = []
    positions = [pos_prev, pos]
    # Minimization loop
    for i in range(10000):
        # Update position
        pos = (2 * pos) - positions[-2] + (DT**2 * gradient(pos))
        positions.append(pos)
        Epot.append(energie(pos))


    plt.figure()
    plt.xlabel('ts')
    plt.ylabel('E')
    plt.title('Energie')
    plt.plot(range(len(Epot)), Epot, label="E_potential")
    plt.legend(loc="upper left")
