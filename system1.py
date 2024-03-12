import numpy as np
import matplotlib.pyplot as plt

d = 0.01
λ = 0.1

l0 = 0.9572
k = 450.0

pos = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0]
])

def energie(pos: np.ndarray):
    return (k / 2) * (distance(pos) - l0)**2

def distance(pos: np.ndarray):
    return np.sqrt( (( pos[0] - pos[1])**2).sum())


def gradient(pos: np.ndarray):
    Δg = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            pos_p = pos.copy()
            pos_p[i, j] += d

            pos_m = pos.copy()
            pos_m[i, j] -= d

            Δg[i, j] =  (energie(pos_p) - energie(pos_m)) / 2*d
    return -Δg


distances = []
energies = []

for i in range(100):
    pos = pos + λ * gradient(pos)
    distances.append(distance(pos))
    energies.append(energie(pos))

plt.figure()
plt.xlabel('ts')
plt.ylabel('A')
plt.title('Distance')
plt.plot(range(len(distances)), distances)

plt.figure()
plt.xlabel('ts')
plt.ylabel('E')
plt.title('Energie')
plt.plot(range(len(energies)), energies)

plt.figure()
plt.xlim([0, 2.4])
plt.ylim([0,25])
plt.plot(np.arange(0.8, 2.4, 0.01), np.vectorize(lambda x: (k / 2) * (x - l0)**2)(np.arange(0.8, 2.4, 0.01)))
plt.plot(distances, energies)
