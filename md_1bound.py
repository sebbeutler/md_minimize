import numpy as np
import math
import matplotlib.pyplot as plt

# Constants
L0 = 0.9572
KB = 450.0

D = 0.001
DT = 0.001

TEMPINIT = 5

# Function to calculate the distance between two points
def distance(x, y):
    return math.sqrt( x**2 + y**2)

# Function to calculate the total energy
def energie(x, y):
    return KB*(distance(x, y) - L0)**2


# Function to calculate the gradient
def gradient(x, y):
    dx = -(energie(x+D, y) - energie(x-D, y)) / (2*D)
    dy = -(energie(x, y+D) - energie(x, y-D)) / (2*D)
    return (dx, dy)

if __name__ == '__main__':
    # Initial positions
    x_prev = 2.0
    y_prev = 2.0
    
    dx_init, dy_init = gradient(x_prev, y_prev)

    x = x_prev + (TEMPINIT**.5 * DT) + (0.5 * dx_init * DT**2)
    y = y_prev + (TEMPINIT**.5 * DT) + (0.5 * dy_init * DT**2)

    Epot = []
    Ekinx = []
    Ekiny = []

    # Minimization loop
    for i in range(10000):
        # Update position

        dx, dy = gradient(x, y)

        x_new = (2 * x) - x_prev + (DT**2 * dx)
        y_new = (2 * y) - y_prev + (DT**2 * dy)

        vx = (x_new - x_prev) / (2 * DT)
        vy = (y_new - y_prev) / (2 * DT)

        Epot.append(energie(x, y))
        Ekinx.append(.5 * vx**2)
        Ekiny.append(.5 * vy**2)

        x_prev = x
        x = x_new

        y_prev = y
        y = y_new
    
    plt.figure()
    plt.xlabel('ts')
    plt.ylabel('E')
    plt.title('Energie')
    plt.plot(range(len(Epot)), Epot, label="E_potential")
    plt.plot(range(len(Ekinx)), Ekinx, label="E_kinetic")
    plt.legend(loc="upper left")
