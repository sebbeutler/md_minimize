"""Module that perform a molecular dynamics minimization
based on the AMBER force field and using the steepest descent algorithm."""

from itertools import combinations

import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

import time

# Dictionary to store the total execution time for each function
function_times = {}

# Function decorator to measure execution time
def track(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if func.__name__ in function_times:
            function_times[func.__name__] += elapsed_time
        else:
            function_times[func.__name__] = elapsed_time
        return result
    return wrapper

@track
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
    return float(np.linalg.norm(p2 - p1))

@track
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

@track
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


class Atom:
    """Born-Oppenheimer approximation of atoms."""

    def __init__(self, name: str, position: np.ndarray, charge: float = 0.0):
        """
        Initialize an Atom object.

        Parameters
        ==========
        name : str
            Name of the atom
        position : np.ndarray
            Position of the atom
        charge : float, optional
            Charge of the atom, by default 0.0
        """
        self.name = name
        self.pos = position

        self.bounds = set()
        self.angles = set()

        self.charge = charge


class System:
    """Perform a steepest-descent minimization from a list of atoms."""

    # List of constants and parameters of the forces calculation
    const = {
        'd': 0.001,
        'dt': 0.001,
        'λ': 0.0001,

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
        """
        Initialize a System object.

        Parameters
        ==========
        **kwargs
            Arbitrary keyword arguments.
        """
        self.atoms_positions: np.ndarray = np.array([]) # Matrix of atoms positions in the system
        self.atoms_name: dict[int, str] = {} # Stores all atoms names
        self.atoms_distances = np.array([]) # matric of distances
        self.bounds: set[frozenset[int]] = set() # stores all bounded atoms
        self.angles: set[tuple[int, ...]] = set() # stores all angles
        self.unbounds = np.array([]) # Triangular matrix of energies between atoms

        self.const.update(kwargs) # Modify the constants from args

        self.step = 0
        self.reset()

    @property
    def natoms(self) -> int:
        return len(self.atoms_positions)

    def reset(self):
        """Reset the system."""
        self.atoms_distances = np.array([])
        self.atoms_epsilon = np.zeros((self.natoms, self.natoms))
        self.atoms_rmin = np.zeros((self.natoms, self.natoms))

        for i, j in combinations(range(self.natoms), 2):
            self.atoms_epsilon[i, j], self.atoms_rmin = self.const['vdw'].get(frozenset((
                self.atoms_name[i],
                self.atoms_name[j])))

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
        self.metrics['coordinates'].append(self.atoms_positions.copy())
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

    def add_atom(self, name: str, position: np.ndarray):
        """
        Add a new atom to the system.

        Parameters
        ==========
        position : np.ndarray
            Position of the atom.

        Returns
        =======
        Atom
            The added atom.
        """
        if self.atoms_positions.size == 0:
            self.atoms_positions = np.array([position])
        else:
            self.atoms_positions = np.vstack([self.atoms_positions, position])

        index = self.natoms-1
        self.atoms_name[index] = name
        return index

    def add_bound(self, atom1: int, atom2: int):
        """
        Add a new bounded interaction between 2 atoms.

        Parameters
        ==========
        atom1 : int
            Index of the first atom.
        atom2 : int
            Index of the second atom.
        """
        bound = frozenset((atom1, atom2))
        self.bounds.add(bound)

    def add_angle(self, atom1: int, atom2: int, atom3: int):
        """
        Add a new angular force between 3 atoms.

        Parameters
        ==========
        atom1 : int
            Index of the first atom.
        atom2 : int
            Index of the second atom.
        atom3 : int
            Index of the third atom.
        """
        angle = (atom1, atom2, atom3)
        self.angles.add(angle)

    @track
    def gradient(self) -> np.ndarray:
        """
        Calculate the gradient of the energy at the position of a given atom.

        This method calculates the gradient of the energy at the position of
        a given atom by using a finite difference approximation.
        The gradient is calculated for each coordinate of the atom's position.

        Parameters
        ----------
        atom : int
            The index of the atom for which to calculate the energy gradient.

        Returns
        -------
        np.ndarray
            The gradient of the energy at the atom's position.
        """
        d = self.const['d']
        g = np.zeros_like(self.atoms_positions)
        atoms_count = self.atoms_positions.shape[0]
        dim_count = self.atoms_positions.shape[1]
        for atom in range(atoms_count):
            for pos in range(dim_count):
                coord = self.atoms_positions[atom, pos]
                self.atoms_positions[atom, pos] = coord + d
                energy_plus = self.energy_total()
                self.atoms_positions[atom, pos] = coord - d
                energy_minus = self.energy_total()
                self.atoms_positions[atom, pos] = coord

                g[atom, pos] = -(energy_plus - energy_minus) / (2*d)
        self.metrics['gradients'][-1].append(g)
        return g

    @track
    def step_minimize(self):
        """
        Update the positions of the atoms along the gradient.
        """
        self.step += 1
        self.compute_distances()
        λ = self.const['λ']
        self.atoms_positions += λ * self.gradient()

        # self.update_metrics()

    def minimize(
        self,
        reset: bool = True,
        max_steps: int=0,
        stop_criteria='GRMS',
        threshold: float=0.1):
        """
        Minimize the energy of the system.

        This method minimizes the energy of the system by performing a series of
        minimization steps. The minimization process continues until a specified
        stop criterionis met or the maximum number of steps is reached.

        Parameters
        ----------
        reset : bool, optional
            If True, the system's metrics and step counter are reset before
            the minimization process. Defaults to True.
        max_steps : int, optional
            The maximum number of steps to perform. If 0, the minimization
            process continues indefinitely until the stop criterion is met.
            Defaults to 0.
        stop_criteria : str, optional
            Mesure used to stop the minimization. Defaults to 'GRMS'.
        threshold : float, optional
            Value for the stop criteria threshold.
        """
        # rest of the function code...
        if reset:
            self.step = 0
            self.reset()
            self.step_minimize()

        mesure = self.metrics[stop_criteria]
        while True:
            self.step_minimize()
            # print(self.step)

            if len(mesure) > 0 and mesure[-1] < threshold:
                break
            if self.step >= max_steps and max_steps != 0:
                break

    def compute_distances(self):
        if self.atoms_distances.size == 0:
            self.atoms_distances = np.zeros((self.natoms, self.natoms), dtype=np.float32)

        for i, j in combinations(range(self.natoms), 2):
            self.atoms_distances[i, j] = distance(
                self.atoms_positions[i],
                self.atoms_positions[j]
            )
        self.atoms_distances += self.atoms_distances.T

    @track
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

    @track
    def energy_bound(self, bound: frozenset[int]) -> float:
        """
        Calculate the energy of a bond between two atoms.

        Parameters
        ----------
        bound : frozenset[int]
            A unique set of 2 atom indices linked by a covalent bound.

        Returns
        -------
        float
            The energy of the bond.
        """
        atom1, atom2 = bound
        l_ideal, k = self.const['bounds'].get(frozenset((
            self.atoms_name[atom1],
            self.atoms_name[atom2])))
        l_actual = self.atoms_distances[atom1, atom2]
        return k * (l_actual - l_ideal)**2

    @track
    def energy_angle(self, angle: tuple[int, ...]) -> float:
        """
        Calculate the energy of an angle formed by three atoms.

        Parameters
        ----------
        angle : tuple[int, ...]
            A unique set of 3 atom indices forming an angle.

        Returns
        -------
        float
            The energy of the angle.
        """
        atom1, atom2, atom3 = angle
        v1 = self.atoms_positions[atom1] - self.atoms_positions[atom2]
        v2 = self.atoms_positions[atom3] - self.atoms_positions[atom2]
        θ, k = self.const['angles'].get(frozenset((
            self.atoms_name[atom1],
            self.atoms_name[atom2],
            self.atoms_name[atom3])))
        return k * (vector_angle(v1, v2, True) - θ)**2

    @track
    def energy_vdw(self) -> np.ndarray:
        """
        Calculate the Van Der Waals energy between two atoms.

        Parameters
        ----------
        atom1 : int
            Index of the first atom.
        atom2 : int
            Index of the second atom.

        Returns
        -------
        float
            The van der Waals energy between the two atoms.
        """
        return self.atoms_epsilon * ((self.atoms_rmin / self.atoms_distances)**12 - 2*(self.atoms_rmin / self.atoms_distances)**6)

    @track
    def energie_coulomb(self, atom1: int, atom2: int) -> float:
        """
        Calculate the electrostatic energy between two atoms.

        Parameters
        ----------
        atom1 : int
            Index of the first atom.
        atom2 : int
            Index of the second atom.

        Returns
        -------
        float
            The Coulomb energy between the two atoms.
        """
        r = self.atoms_distances[atom1, atom2]
        q1 = self.const['charges'][self.atoms_name[atom1]]
        q2 = self.const['charges'][self.atoms_name[atom2]]
        return self.const['ke'] * (q1 * q2) / r

    @track
    def energy_unbound(self) -> float:
        """Compute all unbounded interactions (VdW + Electrostatic).

        Returns
        -------
        float
            Sum of all unbounded interactions
        """
        N = self.natoms
        if self.unbounds.size == 0:
            self.unbounds = np.zeros((2, N, N), dtype=np.float32)

        for atom1, atom2 in combinations(range(N), 2):
            if frozenset((atom1, atom2)) in self.bounds:
                continue
            self.unbounds[1, atom1, atom2] = self.energie_coulomb(atom1, atom2)
        self.unbounds[0] = self.energy_vdw()
        return self.unbounds.sum()

    def plot(self, ax=None, mesure='energies', xlabel='', ylabel='', **kwargs):
        """Plot a measure

        Parameters
        ==========
        mesure : str, optional
            Measure to plot, by default 'energies'
        xlabel : str, optional
            Label for the x-axis, by default ''
        ylabel : str, optional
        """
        if ax is None:
            ax = plt
        ax.grid(True)
        ax.set_xlabel(xlabel)   # type: ignore
        ax.set_ylabel(ylabel)   # type: ignore
        nb_mesure = range(len(self.metrics[mesure]))
        ax.plot(nb_mesure, self.metrics[mesure], **kwargs)

    def plot_all(self, dpi=150.0, width=9, height=5):
            """Plot all metrics in a grid of subplots.

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
                l = ax.figure.subplotpars.left  # type: ignore
                r = ax.figure.subplotpars.right # type: ignore
                t = ax.figure.subplotpars.top   # type: ignore
                b = ax.figure.subplotpars.bottom    # type: ignore
                figw = float(w)/(r-l)
                figh = float(h)/(t-b)
                ax.figure.set_size_inches(figw, figh)   # type: ignore

            fig, axes = plt.subplots(2, 3)
            fig.tight_layout()
            _set_size(9, 5)
            for i, mesure in enumerate(self.metrics):
                if mesure == 'coordinates':
                    continue
                self.plot(axes[i%2][i%3], mesure)
            plt.show()

    def save(self, filename='out.pdb'):
        """
        Save all the atoms coordinate for each step into a PDB file.

        Parameters
        ----------
        filename : str, optional
            The name of the file system  will be saved. Defaults to 'out.pdb'.
        """
        with open(filename, 'w') as pdb:
            pdb.write("HEADER SYSTEM1\n")
            pdb.write("TITLE MINIMIZE 2 BOUNDS\n")

            for i, frame in enumerate(self.metrics['coordinates']):
                pdb.write(f'MODEL {i}\n')

                # Atoms coordinates
                for atom in range(frame.shape[0]):
                    pdb.write(
                        "{:6s}{:5d} {:^4s}{:1s}{:3s} "
                        "{:1s}{:4d}{:1s}    "
                        "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}              "
                        "{:>2s}{:2s}"
                        "\n"
                        .format(
                            "ATOM", atom+1, 'H', ' ', 'H20', 'A', 1, ' ',
                            frame[atom, 0], frame[atom, 1],
                            frame[atom, 2], 1.0, 1.0, 'H', '  '
                        )
                    )

                # Covalent bounds from atoms indices
                for bound in self.bounds:
                    atom1, atom2 = bound
                    pdb.write(
                        "CONECT{:5d}{:5d}\n"
                        .format(
                            atom1+1,
                            atom2+1
                        )
                    )
                pdb.write('ENDMDL\n')
            pdb.write('END\n')

    #TODO
    def load(self, filename: str):
        pass
