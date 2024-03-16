from itertools import combinations
from typing import Any, Callable

import numpy as np

# Function to calculate the distance between two points
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(((p1 - p2)**2).sum())


# Function to calculate the angle between two vectors
def vector_angle(v1, v2, degrees=False) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    if degrees:
        return np.degrees(np.arccos(cosine_angle))

    return np.arccos(cosine_angle)


def rmsd(mobile: np.ndarray, ref: np.ndarray):
    atom_count = ref.shape[0]
    result = 0
    for atom in range(atom_count):
        result += ((mobile[atom] - ref[atom])**2).sum()
    result /= atom_count
    return np.sqrt(result)


class Atom:
    def __init__(self, name: str, position: np.ndarray, charge: float = 0.0):
        self.name = name
        self.pos = position
        self.pos_prev = position.copy()

        self.bounds = set()
        self.angles = set()

        self.charge = charge


class System:
    params = {
        'd': 0.001,
        'dt': 0.001,
        'λ': 0.0001,
        'temperature': 5.0,

        # (l, k)
        'bounds': {
            frozenset(('H', 'O')): (0.9572, 450.0),
        },

        # (θ, k)
        'angles': {
            frozenset(('H', 'O', 'H')): (104.52, 0.016764),
        },

        # (epsilon, rMin)
        'vdw': {
            frozenset(('H', 'H')): (0.046, 0.449),
            frozenset(('H', 'O')): (0.0836, 1.9927),
            frozenset(('O', 'O')): (0.152, 3.5364),
        },

        'charges': {
            'H':  0.417,
            'O': -0.834,
        },

        'ke': 332.0716
    }

    def __init__(self, **kwargs):
        self.atoms: list[Atom] = []
        self.bounds: set[frozenset[Atom]] = set()
        self.angles: set[tuple[Atom, ...]] = set()
        self.unbounds = None

        self.params.update(kwargs)

        self.step = 0
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics: dict[str, Any] = {
            'coordinates': [],
            'energies': [],
            'RMSD': [0.0],
            'energie_diff': [0.0],
            'max_gradients': [],
            'GRMS': [],
        }

    def update_metrics(self):
        self.metrics['coordinates'].append(self.atoms_coordinates())
        self.metrics['energies'].append(self.energy_total())

        if len(self.metrics['coordinates']) >= 2:
            self.metrics['RMSD'].append(
                rmsd(self.metrics['coordinates'][-1], self.metrics['coordinates'][-2]))
        if len(self.metrics['energies']) >= 2:
            self.metrics['energie_diff'].append(
                self.metrics['energies'][-2] - self.metrics['energies'][-1])

        gradients = np.array(
            [self.gradient(atom, self.energy_total) for atom in self.atoms])
        self.metrics['max_gradients'].append(gradients.max())
        self.metrics['GRMS'].append(
            np.sqrt((gradients**2).sum() / gradients.size))

    def add_atom(self, name: str, position: np.ndarray) -> Atom:
        atom = Atom(name, position)
        self.atoms.append(atom)
        return atom

    def add_bound(self, atom1: Atom, atom2: Atom):
        bound = frozenset((atom1, atom2))
        self.bounds.add(bound)
        atom1.bounds.add(bound)
        atom2.bounds.add(bound)

    def add_angle(self, atom1: Atom, atom2: Atom, atom3: Atom):
        angle = (atom1, atom2, atom3)
        self.angles.add(angle)
        atom1.angles.add(angle)
        atom2.angles.add(angle)

    def atoms_coordinates(self) -> np.ndarray:
        coords = np.zeros(
            (len(self.atoms), self.atoms[0].pos.size), np.float32)
        for i, atom in enumerate(self.atoms):
            coords[i] = atom.pos.copy()
        return coords

    ##############
    #  GRADIENT  #
    ##############

    def gradient(self, atom: Atom, f: Callable):
        d = self.params['d']
        g = np.zeros_like(atom.pos)
        for i, coord in enumerate(atom.pos):
            atom.pos[i] = coord + d
            energy_plus = self.energy_total()
            atom.pos[i] = coord - d
            energy_minus = self.energy_total()
            atom.pos[i] = coord

            g[i] = -(energy_plus - energy_minus) / (2*d)
        return g

    def step_minimize(self):
        λ = self.params['λ']
        for atom in self.atoms:
            atom.pos = atom.pos + λ * self.gradient(atom, self.energy_total)

        self.update_metrics()

    #TODO: stop criteria
    def minimize(self, reset: bool = True):
        if reset:
            self.step = 1
            self.reset_metrics()
            self.step_minimize()

        while True:
            self.step += 1
            self.step_minimize()

            if self.metrics['RMSD'][-1] < 0.01:
                break

    def step_md(self, initial=False):
        dt = self.params['dt']
        T = self.params['temperature']
        self.energy_unbound()

        for atom in self.atoms:
            g = self.gradient(atom, self.energy_atom)
            if not initial:
                atom.pos_prev = (2*atom.pos) - atom.pos_prev + (dt**2 * g)
            else:
                atom.pos_prev = atom.pos + (T**.5 * dt) + (0.5 * g * dt**2)

        for atom in self.atoms:
            atom.pos, atom.pos_prev = (atom.pos_prev, atom.pos)

    ###########
    #  ENERGY #
    ###########

    def energy_total(self) -> float:
        energies_bounds = sum(map(self.energy_bound, self.bounds))
        energies_angles = sum(map(self.energy_angle, self.angles))
        energies_unbound = self.energy_unbound()

        return energies_bounds + energies_angles + energies_unbound

    def energy_atom(self, atom: Atom) -> float:
        energies_bounds = sum(map(self.energy_bound, atom.bounds))
        energies_angles = sum(map(self.energy_angle, atom.angles))
        energies_unbound = 0.0

        if not self.unbounds is None:
            id = self.atoms.index(atom)
            energies_unbound = self.unbounds[:, id, :].sum(
            ) + self.unbounds[:, :, id].sum()

        return energies_bounds + energies_angles + energies_unbound

    def energy_bound(self, bound: frozenset[Atom]) -> float:
        atom1, atom2 = bound
        l, k = self.params['bounds'].get(frozenset((atom1.name, atom2.name)))
        return k * (distance(atom1.pos, atom2.pos) - l)**2

    def energy_angle(self, angle: tuple[Atom, ...]) -> float:
        atom1, atom2, atom3 = angle
        v1 = atom1.pos - atom2.pos
        v2 = atom3.pos - atom2.pos
        θ, k = self.params['angles'].get(
            frozenset((atom1.name, atom2.name, atom3.name)), (0, 0))
        return k * (vector_angle(v1, v2, True) - θ)**2

    def energy_vdw(self, atom1: Atom, atom2: Atom) -> float:
        r = distance(atom1.pos, atom2.pos)
        epsilon, rmin = self.params['vdw'].get(
            frozenset((atom1.name, atom2.name)), (0, 0))
        return 4 * epsilon * ((rmin / r)**12 - (rmin / r)**6)

    # TODO formule? see: https://en.wikipedia.org/wiki/AMBER
    def energie_coulomb(self, atom1: Atom, atom2: Atom):
        return self.params['ke'] * (atom1.charge * atom2.charge) / distance(atom1.pos, atom2.pos)

    def energy_unbound(self):
        natoms = len(self.atoms)
        if self.unbounds is None:
            self.unbounds = np.zeros((2, natoms, natoms), dtype=np.float32)

        for (i, atom1), (j, atom2) in combinations(enumerate(self.atoms), 2):
            if frozenset((atom1, atom2)) in atom1.bounds:
                continue
            self.unbounds[0, i, j] = self.energy_vdw(atom1, atom2)
            self.unbounds[1, i, j] = self.energie_coulomb(atom1, atom2)
        return self.unbounds.sum()

    #########
    #  GUI  #
    #########

    def _toggle_minimize(self):
        if self.is_minimizing:
            self.is_minimizing = False
            self.btn_minimize.color = (120, 170, 172)
        else:
            self.is_minimizing = True
            self.btn_minimize.color = (164, 202, 202)

    def _toggle_simulate(self):
        if self.is_simulating:
            self.is_simulating = False
            self.btn_md.color = (120, 170, 172)
        else:
            self.step_md(initial=True)
            self.is_simulating = True
            self.btn_md.color = (164, 202, 202)

    def _reset_playground(self):
        for i, atom in enumerate(self.atoms):
            atom.pos = self.initial_coords[i]

        self.is_minimizing = False
        self.is_simulating = False
        self.btn_md.color = (120, 170, 172)
        self.btn_minimize.color = (120, 170, 172)

    def playground(self):
        import mdsystem_gui as gui

        self.initial_coords = self.atoms_coordinates()

        self.is_minimizing = False
        self.btn_minimize = gui.Button(pos=(0, 0), text="minimize")
        self.btn_minimize.callback = self._toggle_minimize

        self.is_simulating = False
        self.btn_md = gui.Button(pos=(100, 0), text="simulate")
        self.btn_md.callback = self._toggle_simulate

        self.btn_reset = gui.Button(pos=(200, 0), text="reset")
        self.btn_reset.callback = self._reset_playground

        gui.buttons = [self.btn_minimize, self.btn_md, self.btn_reset]

        gui.init()
        while True:
            if self.is_minimizing:
                self.step_minimize()

            if self.is_simulating:
                self.step_md()

            gui.update(self)
            gui.draw(self)

    ###########
    #  UTILS  #
    ###########

    def plot(self, mesure='energies', xlabel='', ylabel='', **kwargs):
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(range(len(self.metrics[mesure])), self.metrics[mesure], **kwargs) 
        plt.show()

    def save(self, filename='out.pdb'):
        with open(filename, 'w') as pdb:
            pdb.write("HEADER SYSTEM1\n")
            pdb.write("TITLE MINIMIZE 2 BOUNDS\n")
            for i, frame in enumerate(self.metrics['coordinates']):
                pdb.write(f'MODEL {i}\n')
                for atom in range(frame.shape[0]):
                    pdb.write(
    "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   \
    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          \
    {:>2s}{:2s}\n".format(
                        "ATOM", atom+1, self.atoms[atom].name, ' ', 'H20', 'A', 1, ' ',
                        frame[atom, 0], frame[atom, 1], frame[atom, 2], 1.0,
                        1.0, self.atoms[atom].name,'  '
                    ))
                for bound in self.bounds:
                    atom1, atom2 = bound
                    pdb.write("CONECT{:5d}{:5d}\n".format(self.atoms.index(atom1)+1, self.atoms.index(atom2)+1))
                pdb.write('ENDMDL\n')
            pdb.write('END\n')

    #TODO
    def load(self, filename: str):
        pass
