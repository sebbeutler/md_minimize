import numpy as np

import matplotlib.pyplot as plt

# Constants
L0 = 0.9572
KB = 450.0

D = 0.001
DT = 0.001

# Function to calculate the distance between two points
def distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt( (( p1 - p2)**2).sum())


# Function to calculate the bound energy
def energie_bound(p1: np.ndarray, p2: np.ndarray, k, l):
    return k*(distance(p1, p2) - l)**2


# Function to calculate the total energy
def energie(pos: np.ndarray, atom: int):
    other_atoms = [0, 1, 2]
    other_atoms.remove(atom)

    return energie_bound(pos[atom], pos[other_atoms[0]], KB, L0) + \
        energie_bound(pos[atom], pos[other_atoms[1]], KB, L0)


# Function to calculate the gradient
def gradient(pos: np.ndarray):
    Δg = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            pos_p = pos.copy()
            pos_p[i, j] += D
            pos_m = pos.copy()
            pos_m[i, j] -= D

            Δg[i, j] =  -(energie(pos_p, i) - energie(pos_m, i)) / (2*D)
    return Δg

if __name__ == '__main__':
    # Initial positions
    pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [-1.7, -2.5, 0.0],
    ])

    # Mesurements
    dist_C1_C2 = []
    dist_C1_C3 = []
    dist_C2_C3 = []

    Epot = []
    Ekin = []
    Etot = []

    pos_init = pos + (5**.5 * DT) + (0.5 * gradient(pos) * DT**2)

    positions: list[np.ndarray] = [pos, pos_init]
    pos = pos_init

    # Minimization loop
    for i in range(10000):
        # Update position

        pos = (2 * pos) - positions[-2] + (DT**2 * gradient(pos))
        positions.append(pos)

        dist_C1_C2.append(distance(pos[0], pos[1]))
        dist_C1_C3.append(distance(pos[0], pos[2]))
        dist_C2_C3.append(distance(pos[1], pos[2]))

        # epot = energie(pos, 0) + energie(pos, 1) + energie(pos, 2)
        # ekin = (0.5*( (pos - positions[-1]) / (2 * DT) )**2).sum()
        epot = energie(pos, 0)
        ekin = (( (positions[-1][0] - positions[-2][0]) / DT )**2).sum()

        Epot.append(epot)
        Ekin.append(ekin)
        Etot.append(epot + ekin)

    # Plotting
    plt.figure()
    plt.xlabel('ts')
    plt.ylabel('A')
    plt.title('Distance')
    plt.plot(range(len(dist_C1_C2)), dist_C1_C2)
    plt.plot(range(len(dist_C1_C3)), dist_C1_C3)
    plt.plot(range(len(dist_C2_C3)), dist_C2_C3)

    plt.figure()
    plt.xlabel('ts')
    plt.ylabel('E')
    plt.title('Energie')
    plt.plot(range(len(Epot)), Epot, label="E_potential")
    plt.plot(range(len(Ekin)), Ekin, label="E_kinetic")
    plt.plot(range(len(Etot)), Etot, label="E_total")
    plt.legend(loc="upper left")
