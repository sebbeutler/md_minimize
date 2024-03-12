import numpy as np
import matplotlib.pyplot as plt


d = 0.1
λ = 0.01

l0 = 1.5
kb = 100

θ = 104
ka = 0.784


def angle(v1, v2, degrees=False):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    if degrees:
        return np.degrees(np.arccos(cosine_angle))

    return np.arccos(cosine_angle)


def energie_angle(v1: np.ndarray, v2: np.ndarray, k, theta):
    return k*(angle(v1, v2, True) - theta)**2


def energie_bound(p1: np.ndarray, p2: np.ndarray, k, l):
    return k*(distance(p1, p2) - l)**2


def energie(pos: np.ndarray):
    return energie_angle(pos[1]-pos[0], pos[2]-pos[0], ka, θ) + \
        energie_bound(pos[0], pos[1], kb, l0) + \
        energie_bound(pos[0], pos[2], kb, l0)
    # return energie_bound(pos[0], pos[1], kb, l0) + \
    #     energie_bound(pos[0], pos[2], kb, l0)


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


def distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt( (( p1 - p2)**2).sum())


def save_pdb(positions):
    with open('system2.pdb', 'w') as pdb:
        pdb.write("HEADER SYSTEM1\n")
        pdb.write("TITLE MINIMIZE 2 BOUNDS\n")
        for i, frame in enumerate(positions):
            pdb.write(f'MODEL {i}\n')
            for atom in range(frame.shape[0]):
                pdb.write(f'ATOM  {atom:5} {names[atom]:^4} H2O A   1    {frame[atom, 0]:8.3f}{frame[atom, 1]:8.3f}{frame[atom, 2]:8.3f}  1.00  1.00           {names[atom][0]}  \n')
            pdb.write('ENDMDL\n')
        pdb.write('END\n')


if __name__ == '__main__':
    names = ['O', 'H1', 'H2']

    pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [-1.7, -2.5, 0.0],
    ])

    # pos = np.array([(1.5, 2.0), (3.0, 2.5), (5.0, 1.7)])

    distances1 = []
    distances2 = []
    energies = []
    angles = []
    positions: list[np.ndarray] = []

    for i in range(1000):
        pos = pos + λ * gradient(pos)
        angles.append(angle(pos[0]-pos[1], pos[0]-pos[2], True))
        positions.append(pos)
        distances1.append(distance(pos[0], pos[1]))
        distances2.append(distance(pos[0], pos[2]))
        energies.append(energie(pos))

    save_pdb(positions)

    plt.figure()
    plt.xlabel('ts')
    plt.ylabel('A')
    plt.title('Distance')
    plt.plot(range(len(distances1)), distances1)
    plt.plot(range(len(distances2)), distances2)

    plt.figure()
    plt.xlabel('ts')
    plt.ylabel('E')
    plt.title('Energie')
    plt.ylim([0,20])
    plt.plot(range(len(energies)), energies)

    plt.figure()
    plt.xlabel('ts')
    plt.ylabel('θ')
    plt.ylim([0,200])
    plt.title('Angle')
    plt.plot(range(len(angles)), angles)

    # plt.figure()
    # plt.xlim([0, 2.4])
    # plt.ylim([0,25])
    # plt.plot(np.arange(0.8, 2.4, 0.01), np.vectorize(lambda x: (k / 2) * (x - l0)**2)(np.arange(0.8, 2.4, 0.01)))
    # plt.plot(distances, energies)
