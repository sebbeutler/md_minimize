from typing import Self
import numpy as np

import system_gui as gui

# Function to calculate the distance between two points
def distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt( (( p1 - p2)**2).sum())


# Function to calculate the angle between two vectors
def vector_angle(v1, v2, degrees=False):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    if degrees:
        return np.degrees(np.arccos(cosine_angle))

    return np.arccos(cosine_angle)


class Atom:
    def __init__(self, name: str, position: np.ndarray):
        self.name = name
        self.pos = position
        self.pos_prev = position.copy()

        self.bounds = set()
        self.angles = set()


class System:
    def __init__(self):
        self.atoms: list[Atom] = []
        self.bounds: set[frozenset[Atom]] = set()
        self.angles: set[tuple[Atom, ...]] = set()
        self.params = {
            'd': 0.001,
            'dt': 0.001,
            'λ': 0.0001,

            'l0': 0.9572,
            'kb': 450.0,

            'θ0': 104.52,
            'ka': 0.016764,

            'temperature': 5.0,
            'epsilonHH' : 0.046,
            'epsilonOH' : 0.0836,
            'epsilonOO' : 0.1521,

            'rminHH' : 0.449,
            'rminOH' : 1.9927,
            'rminOO' : 3.5364,
            'q1': 0.417,
            'q2': -0.834,
            'ke': 332.0716
            }

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

    def step_minimize(self):
        d = self.params['d']
        λ = self.params['λ']
        for atom in self.atoms:
            for i, coord in enumerate(atom.pos):
                atom.pos[i] = coord + d
                energy_plus = self.energy_total()
                atom.pos[i] = coord - d
                energy_minus = self.energy_total()

                gradient = -(energy_plus - energy_minus) / (2*d)
                atom.pos[i] = coord + λ * gradient

    def step_md(self, initial=False):
        d = self.params['d']
        dt = self.params['dt']
        T = self.params['temperature']
        for atom in self.atoms:
            for i, coord in enumerate(atom.pos):
                atom.pos[i] = coord + d
                energy_plus = self.energy_atom(atom)
                atom.pos[i] = coord - d
                energy_minus = self.energy_atom(atom)
                atom.pos[i] = coord

                gradient = -(energy_plus - energy_minus) / (2*d)
                if not initial:
                    atom.pos_prev[i] = (2*coord) - atom.pos_prev[i] + (dt**2 * gradient)
                else:
                    atom.pos_prev[i] = coord + (T**.5 * dt) + (0.5 * gradient * dt**2)

        for atom in self.atoms:
            atom.pos, atom.pos_prev = (atom.pos_prev, atom.pos)

    def energy_total(self) -> float:
        energy_bounds = sum(map(self.energy_bound, self.bounds))
        energy_angles = sum(map(self.energy_angle, self.angles))
        return energy_bounds + energy_angles

    def energy_atom(self, atom : Atom) -> float:
        energies_bounds = sum(map(self.energy_bound, atom.bounds))
        energies_angles = sum(map(self.energy_angle, atom.angles))
        energies_vdw = 0.0
        for target_atom in self.atoms:
            if atom == target_atom:
                continue
            if frozenset((atom, target_atom)) in atom.bounds:
                continue
        return energies_bounds + energies_angles

    def energy_bound(self, bound: frozenset[Atom]) -> float:
        atom1, atom2 = bound
        return self.params['kb'] * (distance(atom1.pos, atom2.pos) - self.params['l0'])**2

    def energy_angle(self, angle: tuple[Atom, ...]) -> float:
        atom1, atom2, atom3 = angle 
        v1 = atom1.pos - atom2.pos
        v2 = atom3.pos - atom2.pos
        return self.params['ka'] * (vector_angle(v1, v2, True) - self.params['θ0'])**2
    
    def energie_vdw(self, atom1: Atom, atom2: Atom, e: float, rmin: float) -> float:
        r = distance(atom1.pos, atom2.pos)
        return 4 * e * ((rmin / r)**12 - (rmin / r)**6)

    def energie_coulomb(self, atom1: Atom, atom2: atom):
        return self.params['ke'] * (atom1.charge * atom2.charge) / distance(atom1.pos, atom2.pos)

    def toggle_minimize(self):
        if self.minimize:
            self.minimize = False
            self.btn_minimize.color = (120,170,172)
        else:
            self.minimize = True
            self.btn_minimize.color = (164,202,202)

    def toggle_simulate(self):
        if self.simulate:
            self.simulate = False
            self.btn_md.color = (120,170,172)
        else:
            self.simulate = True
            self.btn_md.color = (164,202,202)
            self.step_md(True)

    def run(self):
        self.minimize = False
        self.simulate = False

        self.btn_minimize = gui.Button(pos=(0,0), text="minimize")
        self.btn_minimize.callback = self.toggle_minimize
        self.btn_md = gui.Button(pos=(100, 0), text="simulate")
        self.btn_md.callback = self.toggle_simulate

        gui.buttons = [self.btn_minimize, self.btn_md]

        gui.init()
        while True:
            if self.minimize:
                self.step_minimize()

            if self.simulate:
                self.step_md()

            gui.update(self)
            gui.draw(self)

