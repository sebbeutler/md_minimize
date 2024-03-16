# -*- coding: utf-8 -*-

import numpy as np

# Constants
d = 0.01
λ = 0.1

l0 = 0.9572
kb = 450.0
d
θ = 104.52
ka = 0.016764

# Function to calculate the distance between two points
def distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt( (( p1 - p2)**2).sum())


# Function to calculate the angle between two vectors
def angle(v1, v2, degrees=False):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    if degrees:
        return np.degrees(np.arccos(cosine_angle))

    return np.arccos(cosine_angle)


# Function to calculate the angle energy
def energie_angle(v1: np.ndarray, v2: np.ndarray, k, theta):
    return k*(angle(v1, v2, True) - theta)**2


# Function to calculate the bound energy
def energie_bound(p1: np.ndarray, p2: np.ndarray, k, l):
    return k*(distance(p1, p2) - l)**2


# Function to calculate the total energy
def energie(pos: np.ndarray):
    return energie_angle(pos[1]-pos[0], pos[2]-pos[0], ka, θ) + \
        energie_bound(pos[0], pos[1], kb, l0) + \
        energie_bound(pos[0], pos[2], kb, l0)


# Function to calculate the gradient
def gradient(pos: np.ndarray):
    Δg = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            pos_p = pos.copy()
            pos_p[i, j] += d
            pos_m = pos.copy()
            pos_m[i, j] -= d

            Δg[i, j] =  -(energie(pos_p) - energie(pos_m)) / 2*d
    return Δg


# Function to save the positions in a PDB file
def save_pdb(positions):
    with open('system2.pdb', 'w') as pdb:
        pdb.write("HEADER SYSTEM1\n")
        pdb.write("TITLE MINIMIZE 2 BOUNDS\n")
        for i, frame in enumerate(positions):
            pdb.write(f'MODEL {i}\n')
            for atom in range(frame.shape[0]):
                pdb.write(
"{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   \
{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          \
{:>2s}{:2s}\n".format(
                    "ATOM", atom+1, names[atom], ' ', 'H20', 'A', 1, ' ',
                    frame[atom, 0], frame[atom, 1], frame[atom, 2], 1.0,
                    1.0, names[atom][0],'  '
                ))
            for mol_index in range(frame.shape[0]//3):
                real_index = mol_index*3+1
                pdb.write("CONECT{:5d}{:5d}\n".format(real_index, real_index+1))
                pdb.write("CONECT{:5d}{:5d}\n".format(real_index, real_index+2))
            pdb.write('ENDMDL\n')
        pdb.write('END\n')

def rmsd(pos1, pos2):
    atom_count = pos1.shape[0]
    result = 0
    for atom in range(atom_count):
      result += ((pos1[atom] - pos2[atom])**2).sum()
    result /= atom_count
    return np.sqrt(result)

# Atoms names
names = ['O', 'H1', 'H2']

# Initial positions
pos = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 0.0],
    [-1.7, -2.5, 0.0],
])

# Mesurements
energies = [energie(pos)]
positions: list[np.ndarray] = [pos]

max_gradients = []
energie_diff = []
RMSD = []
GRMS_values = []
gradients = []

step = 800

# Minimization loop
for i in range(step):
    # Update position
    grad = gradient(pos)
    pos = pos + λ * grad

    positions.append(pos)
    energies.append(energie(pos))
    gradients.append(grad)

    max_gradients.append(np.max(grad))
    GRMS_values.append(np.sqrt((grad**2).sum() / grad.size))

    energie_diff.append(energies[-2] - energies[-1])
    RMSD.append(rmsd(positions[-1], positions[-2]))

from itertools import accumulate
# Plotting

import matplotlib.pyplot as plt

steps = np.arange(step)
metrics = [energies[1:], max_gradients, energie_diff, RMSD, GRMS_values, gradients]
metrics_names = [
    "energies", "max_gradients", "energie_diff",
    "RMSD", "GRMS_values", "gradients"]

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

fig.set_dpi(150.0)
fig.tight_layout()
set_size(9, 5)

ax1.set_ylabel('E')
ax1.set_title('Energie')
ax1.grid(True)
ax1.plot(steps, energies[1:])

ax2.set_ylabel('gradient')
ax2.set_title('Max Gradient')
ax2.grid(True)
ax2.plot(steps, max_gradients)

ax3.set_ylabel('GRMS')
ax3.set_title('GRMS')
ax3.grid(True)
ax3.plot(steps, GRMS_values)

ax4.set_ylabel('E')
ax4.set_title('Energie difference')
ax4.grid(True)
ax4.plot(steps, energie_diff)

ax5.set_ylabel('RMSD')
ax5.set_title('RMSD')
ax5.grid(True)
ax5.plot(steps, RMSD)

ax6.set_ylabel('stop criteria')
ax6.set_title('Consensus')
ax6.grid(True)

# gradients = np.array(list(map(lambda m: -np.gradient(m, steps) / max(m), metrics)))
# consensus = np.sqrt((gradients**2).sum(axis=0) / len(metrics))

# ax6.plot(steps, consensus, color="black")

# for grad in gradients:
#   ax6.plot(steps, grad, linestyle=(0, (1, 1)))


# for i in range(len(metrics)-1):
#     for j in range(i, len(metrics)):
#         ax6.plot(steps, np.degrees(np.arctan2(np.gradient(metrics[i], steps), np.gradient(metrics[j], steps))))
# ax6.plot(steps, np.degrees(np.arctan2(-np.gradient(energies[1:], steps), 1)))
ax6.plot(steps, np.degrees(np.arctan2(-np.gradient(max_gradients, steps), 1)))

from sklearn.metrics.pairwise import cosine_similarity

def fluctuation(mesure, label):
  dx = -np.gradient(mesure, steps)
  # dx = np.array(mesure[1:]) / np.array(mesure[:-1])
  # plt.plot(steps[:-1], dx, label=label)
  dx = np.arctan2(dx, d)
  plt.plot(steps, dx, label=label)
  plt.legend()

# for name, mesure in zip(metrics_names, metrics):
#   fluctuation(mesure, name)

def angle_between_matrices(matrix1, matrix2):
    # Flatten the matrices to 1D arrays
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Calculate angles between consecutive matrices
angles = []
for i in range(len(gradients) - 1):
    matrix1 = gradients[i]
    matrix2 = gradients[i + 1]
    angle = angle_between_matrices(matrix1, matrix2)
    angles.append(angle)

# Plotting
plt.plot(angles)
plt.grid(True)
plt.show()

save_pdb(positions)