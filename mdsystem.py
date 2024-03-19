"""Minimization System in molecula dynamics.

Module that perform a molecular dynamics minimization
based on the AMBER force field and using the steepest descent algorithm.
"""

from itertools import combinations
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    p1 : np.ndarray
        The coordinates of the first point.
    p2 : np.ndarray
        The coordinates of the second point.

    Returns
    -------
    float
        The Euclidean distance between the two points.
    """
    return np.sqrt(((p1 - p2)**2).sum())


def vector_angle(v1, v2, degrees=False) -> float:
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    v1 : array_like
        The first input vector.
    v2 : array_like
        The second input vector.
    degrees : bool, optional
        If True, the angle is returned in degrees.
        Otherwise, the angle is returned in radians.
        Defaults to False.

    Returns
    -------
    float
        The angle between the two vectors in radians or degrees.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    if degrees:
        return np.degrees(np.arccos(cosine_angle))

    return np.arccos(cosine_angle)


def rmsd(mobile: np.ndarray, ref: np.ndarray):
    """
    Calculate the root-mean-square deviation (RMSD) between two sets of points.

    Parameters
    ----------
    mobile : np.ndarray
        The coordinates of the first set of points.
    ref : np.ndarray
        The coordinates of the second set of points.

    Returns
    -------
    float
        The RMSD between the two sets of points.
    """
    atom_count = ref.shape[0]
    result = 0
    for atom in range(atom_count):
        result += ((mobile[atom] - ref[atom])**2).sum()
    result /= atom_count
    return np.sqrt(result)


class Atom:  # pylint: disable=too-few-public-methods
    """Born-Oppenheimer approximation of atoms."""

    def __init__(
        self,
        name: str,
        molecule: str,
        chain : str,
        charge: float,
        position: np.ndarray
    ):
        """
        Initialize an Atom object.

        Parameters
        ==========
        name : str
            Name of the atom.
        molecule : str
            Name of the molecule that contains the atom.
        chain : str
            Chain identifier for the atom.
        charge : float
            Charge of the atom.
        position : np.ndarray
            Position of the atom.
        """
        self.name = name
        self.mol = molecule
        self.chain = chain
        self.pos = position

        self.bounds = set()
        self.angles = set()

        self.charge = charge


class System:
    """Perform a steepest-descent minimization from a list of atoms."""

    # List of constants and parameters of the forces calculation
    const = {
        'd': 0.01,
        'λ': 0.001,

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

        'ke': 332.0716
    }

    def __init__(self, **kwargs):
        """
        Initialize a System object.

        Parameters
        ==========
        **kwargs
            Arbitrary keyword arguments.
        """
        self.atoms: list[Atom] = []  # list of atoms in the system
        self.bounds: set[frozenset[Atom]] = set()  # stores all bounded atoms
        self.angles: set[tuple[Atom, ...]] = set()  # stores all angles
        self.unbounds = None  # Triangular matrix of energies between atoms
        self.distances: dict[frozenset[Atom], float] = {}  # Atoms distances

        self.const.update(kwargs)  # Modify the constants from args

        self.step = 0
        self.reset_metrics()

    def reset_metrics(self):
        """Reset metrics at each minimization step."""
        self.step = 0
        self.metrics: dict[str, list] = {
            'coordinates': [],
            'energies': [],
            'RMSD': [0.0],
            'energie_diff': [0.0],
            'gradients': [[]],
            'max_gradients': [],
            'GRMS': [],
        }

    def update_metrics(self):
        """Record a measure for the current atom positions."""
        # Atoms postion
        self.metrics['coordinates'].append(self.atoms_coordinates())
        # Total system energy
        self.metrics['energies'].append(self.energy_total())
        # RMSD
        if len(self.metrics['coordinates']) >= 2:
            self.metrics['RMSD'].append(rmsd(
                self.metrics['coordinates'][-1],
                self.metrics['coordinates'][-2]))
        # Energy difference
        if len(self.metrics['energies']) >= 2:
            self.metrics['energie_diff'].append(
                self.metrics['energies'][-2] - self.metrics['energies'][-1])
        # Maximum of all gradients
        last_gradients = np.array(self.metrics['gradients'][-1])
        self.metrics['max_gradients'].append(last_gradients.max())
        # GRMS
        grms = np.sqrt((last_gradients**2).sum() / last_gradients.size)
        self.metrics['GRMS'].append(grms)
        self.metrics['gradients'].append([])

    def add_atom(
        self,
        name: str,
        mol: str,
        chain: str,
        charge: float,
        position: np.ndarray
    ) -> Atom:
        """
        Add a new atom to the system.

        Parameters
        ==========
        name : str
            Name of the atom.
        mol : str
            Molecule identifier for the atom.
        chain : str
            Chain identifier for the atom.
        position : np.ndarray
            Position of the atom.

        Returns
        =======
        Atom
            The added atom.
        """
        atom = Atom(name, mol, chain, charge, position)
        self.atoms.append(atom)
        return atom

    def add_bound(self, atom1: Atom, atom2: Atom):
        """
        Add a new bounded interaction between 2 atoms.

        Parameters
        ==========
        atom1 : Atom
            First atom.
        atom2 : Atom
            Second atom.
        """
        bound = frozenset((atom1, atom2))
        self.bounds.add(bound)
        atom1.bounds.add(bound)
        atom2.bounds.add(bound)

    def add_angle(self, atom1: Atom, atom2: Atom, atom3: Atom):
        """
        Add a new angular force between 3 atoms.

        Parameters
        ==========
        atom1 : Atom
            First atom.
        atom2 : Atom
            Second atom.
        atom3 : Atom
            Third atom.
        """
        angle = (atom1, atom2, atom3)
        self.angles.add(angle)
        atom1.angles.add(angle)
        atom2.angles.add(angle)

    def atoms_coordinates(self) -> np.ndarray:
        """
        Return the positions of the atoms as an array.

        Returns
        =======
        np.ndarray
            Array of atom positions.
        """
        coords = np.zeros(
            (len(self.atoms), self.atoms[0].pos.size), np.float32)
        for i, atom in enumerate(self.atoms):
            coords[i] = atom.pos.copy()
        return coords

    def gradient(self, atom: Atom) -> np.ndarray:
        """
        Calculate the gradient of the energy at the position of a given atom.

        This method calculates the gradient of the energy at the position of
        a given atom by using a finite difference approximation.
        The gradient is calculated for each coordinate of the atom's position.

        Parameters
        ----------
        atom : Atom
            The atom for which to calculate the energy gradient.

        Returns
        -------
        np.ndarray
            The gradient of the energy at the atom's position.
        """
        d = self.const['d']
        g = np.zeros_like(atom.pos)
        for i, coord in enumerate(atom.pos):
            atom.pos[i] = coord + d
            self.update_atom_distances(atom)
            energy_plus = self.energy_total()

            atom.pos[i] = coord - d
            self.update_atom_distances(atom)
            energy_minus = self.energy_total()

            atom.pos[i] = coord

            g[i] = -(energy_plus - energy_minus) / (2*d)

        self.update_atom_distances(atom)
        self.metrics['gradients'][-1].append(g)
        return g

    def update_all_distances(self):
        for atom1, atom2 in combinations(self.atoms, 2):
            atom_pair = frozenset((atom1, atom2))
            self.distances[atom_pair] = distance(atom1.pos, atom2.pos)

    def update_atom_distances(self, atom):
        for other_atom in self.atoms:
            if atom == other_atom:
                continue
            atom_pair = frozenset((atom, other_atom))
            self.distances[atom_pair] = distance(atom.pos, other_atom.pos)


    def step_minimize(self):
        """Update the positions of the atoms along the gradient."""
        self.step += 1

        # Update distances
        self.update_all_distances()

        # Update positions
        _lambda = self.const['λ']
        for atom in self.atoms:
            atom.pos = atom.pos + _lambda * self.gradient(atom)

        self.update_metrics()

    def minimize(
        self,
        reset: bool = True,
        max_steps: int = 500,
        min_steps: int = 100,
        stop_criteria: str = 'GRMS',
        threshold: float = 0.1
    ):
        """
        Minimize the energy of the system.

        This method minimizes the energy of the system by performing a series
        of minimization steps. The minimization process continues until the
        stop criteria is met or the maximum number of steps is reached.

        Parameters
        ----------
        reset : bool, optional
            If True, the system's metrics and step counter are reset before
            the minimization process. Defaults to True.
        max_steps : int, optional
            The maximum number of steps to perform. Defaults to 500.
        min_steps : int, optional
            The minimum number of steps to perform. Defaults to 100.
        stop_criteria : str, optional
            Mesure used to stop the minimization. Defaults to 'GRMS'.
        threshold : float, optional
            Value for the stop criteria threshold.
        """
        if reset:
            self.reset_metrics()
            self.metrics['coordinates'].append(self.atoms_coordinates())
            self.step_minimize()

        progress_bar = tqdm(range(max_steps), desc="Minimization: ")
        for _ in progress_bar:
            self.step_minimize()

            mesure = self.metrics[stop_criteria]
            if self.step > min_steps and mesure[-1] < threshold:
                progress_bar.close()
                print(f'Reached stop criteria: {stop_criteria} < {threshold}')
                break
        print('Total steps: ', self.step)

    def energy_total(self) -> float:
        """
        Calculate the total energy of the system.

        This method calculates the total energy of the system by summing up
        the energy of all bonds, angles, and unbound atoms in the system.

        Returns
        -------
        float
            The total energy of the system.
        """
        energies_bounds = sum(map(self.energy_bound, self.bounds))
        energies_angles = sum(map(self.energy_angle, self.angles))
        energies_unbound = self.energy_unbound()

        return energies_bounds + energies_angles + energies_unbound

    def energy_bound(self, bound: frozenset[Atom]) -> float:
        """
        Calculate the energy of a bond between two atoms.

        This method calculates the energy of a bond between two atoms
        based on the bond's actual length and its ideal length.
        The energy is calculated using the formula:
            k * (l_actual - l_ideal)**2,
        where k is the force constant and l is the bond length.

        Parameters
        ----------
        bound : frozenset[Atom]
            A unique set of 2 atoms linked by a covalent bound.

        Returns
        -------
        float
            The energy of the bond.
        """
        atom1, atom2 = bound
        l_ideal, k = self.const['bounds'].get(
            frozenset((atom1.name[0], atom2.name[0])))
        l_actual = self.distances[frozenset((atom1, atom2))]
        return k * (l_actual - l_ideal)**2

    def energy_angle(self, angle: tuple[Atom, ...]) -> float:
        """
        Calculate the energy of an angle formed by three atoms.

        This method calculates the energy of an angle formed by
        three atoms based on the angle's actual value and its ideal value.
        The energy is calculated using the formula:
            k * (θ_actual - θ_ideal)**2,
        where k is the force constant and θ is the angle.

        Parameters
        ----------
        angle : tuple[Atom, ...]
            A unique set of 3 atoms forming an angle.

        Returns
        -------
        float
            The energy of the angle.
        """
        atom1, atom2, atom3 = angle
        v1 = atom1.pos - atom2.pos
        v2 = atom3.pos - atom2.pos
        θ, k = self.const['angles'].get(
            frozenset((atom1.name[0], atom2.name[0], atom3.name[0])))
        return k * (vector_angle(v1, v2, True) - θ)**2

    def energy_vdw(self, atom1: Atom, atom2: Atom) -> float:
        """
        Calculate the Van Der Waals energy between two atoms.

        This method calculates the Van Der Waals energy between two atoms based
        on their distance and the constant parameters for their atom types.
        The energy is calculated using the Lennard-Jones potential formula:
            ε * ((rmin / r)**12 - 2 * (rmin / r)**6),
            where ε is the depth of the potential well,
            rmin is the distance at which the potential reaches its minimum
            and r is the distance between the atoms.


        Args:
            atom1 (Atom): The first atom.
            atom2 (Atom): The second atom.

        Returns:
            float: The van der Waals energy between the two atoms.
        """
        r = self.distances[frozenset((atom1, atom2))]
        epsilon, rmin = self.const['vdw'].get(
            frozenset((atom1.name[0], atom2.name[0])), (0, 0))
        return epsilon * ((rmin / r)**12 - 2*(rmin / r)**6)

    def energie_coulomb(self, atom1: Atom, atom2: Atom) -> float:
        """
        Calculate the electrostatic energy between two atoms.

        This method calculates the electrostatic energy between two atoms
        based their charges and the distance between them.
        Using the Coulomb's law, the energy is calculated with the formula:
            ke * (q1 * q2) / d,
        where ke is Coulomb's constant,
        q1 and q2 are the charges of the atoms
        and d is the distance between the atoms.

        Parameters
        ----------
        atom1 : Atom
            The first atom involved in the interaction.
        atom2 : Atom
            The second atom involved in the interaction.

        Returns
        -------
        float
            The Coulomb energy between the two atoms.
        """
        r = self.distances[frozenset((atom1, atom2))]
        return self.const['ke'] * (atom1.charge * atom2.charge) / r

    def energy_unbound(self) -> float:
        """
        Compute all unbounded interactions (VdW + Electrostatic).

        Returns
        =======
        float
            Sum of all unbounded interactions
        """
        natoms = len(self.atoms)
        if self.unbounds is None:
            self.unbounds = np.zeros((2, natoms, natoms), dtype=np.float32)

        for (i, atom1), (j, atom2) in combinations(enumerate(self.atoms), 2):
            # if atom1.mol + atom1.chain == atom2.mol + atom2.chain:
            if frozenset((atom1, atom2)) in self.bounds:
                continue
            self.unbounds[0, i, j] = self.energy_vdw(atom1, atom2)
            self.unbounds[1, i, j] = self.energie_coulomb(atom1, atom2)
        return self.unbounds.sum()

    def plot(self, ax=None, mesure='energies', xlabel='', ylabel='', **kwargs):
        """
        Plot a measure.

        Parameters
        ==========
        mesure : str, optional
            Measure to plot, by default 'energies'
        xlabel : str, optional
            Label for the x-axis, by default ''
        ylabel : str, optional
        """
        mesure_count = range(len(self.metrics[mesure]))

        if ax is None:
            plt.grid(True)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(mesure)
            plt.plot(mesure_count, self.metrics[mesure], **kwargs)
            plt.show()
        else:
            ax.grid(True)
            ax.set_xlabel(xlabel)   # type: ignore
            ax.set_ylabel(ylabel)   # type: ignore
            ax.set_title(mesure)    # type: ignore
            mesure_count = range(len(self.metrics[mesure]))
            ax.plot(mesure_count, self.metrics[mesure], **kwargs)

    def plot_all(self, dpi=150.0, width=9, height=5):
        """
        Plot all metrics in a grid of subplots.

        Args:
            dpi : float, optional
                Dots per inch for the figure. Defaults to 150.0.
            width : int, optional
                Width of the figure in inches. Defaults to 9.
            height : int, optional
                Height of the figure in inches. Defaults to 5.
        """
        def _set_size(w: int, h: int):
            ax = plt.gca()
            left = ax.figure.subplotpars.left      # type: ignore
            right = ax.figure.subplotpars.right    # type: ignore
            top = ax.figure.subplotpars.top        # type: ignore
            bottom = ax.figure.subplotpars.bottom  # type: ignore
            figw = float(w)/(right-left)
            figh = float(h)/(top-bottom)
            ax.figure.set_size_inches(figw, figh)  # type: ignore

        fig, axes = plt.subplots(2, 3)
        fig.set_dpi(dpi)
        fig.tight_layout()
        _set_size(width, height)
        for i, mesure in enumerate(self.metrics):
            if mesure in ['coordinates', 'gradients']:
                continue
            self.plot(axes[i % 2][i % 3], mesure)
        plt.show()

    def save(self, filename='out.pdb'):
        """
        Save all the atoms coordinate for each step into a PDB file.

        Parameters
        ----------
        filename : str, optional
            The name of the file system  will be saved. Defaults to 'out.pdb'.
        """
        with open(filename, 'w', encoding="utf-8") as pdb:
            pdb.write("HEADER SYSTEM1\n")
            pdb.write("TITLE MINIMIZE 2 BOUNDS\n")

            for i, frame in enumerate(self.metrics['coordinates']):
                pdb.write(f'MODEL {i+1}\n')

                # Atoms coordinates
                for i in range(frame.shape[0]):
                    atom = self.atoms[i]
                    pdb.write(
                        f"{'ATOM':6s}{i+1:5d} {atom.name:^4s}"
                        f"{'':1s}{atom.mol:3s} {atom.chain:1s}{1:4d}"
                        f"{'':1s}   {frame[i, 0]:8.3f}{frame[i, 1]:8.3f}"
                        f"{frame[i, 2]:8.3f}{1.0:6.2f}{1.0:6.2f}          "
                        f"{self.atoms[i].name:>2s}{'':2s}"
                        "\n"
                    )

                # Covalent bounds from atoms indices
                for bound in self.bounds:
                    atom1, atom2 = bound
                    pdb.write(
                        f"CONECT{self.atoms.index(atom1)+1:5d}"
                        f"{self.atoms.index(atom2)+1:5d}\n"
                    )
                pdb.write('ENDMDL\n')
            pdb.write('END\n')
