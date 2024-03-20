import numpy as np
from minimize import System


def add_water_molecule(index: int, position: np.ndarray):
    O = system.add_atom(f'O', 'H20', str(index), -0.834, position)
    H1 = system.add_atom(f'H', 'H20', str(index), 0.417, position + np.array([-1.0, 1.0, 0.5]))
    H2 = system.add_atom(f'H', 'H20', str(index), 0.417, position + np.array([1.0, 1.0, 0.5]))

    system.add_bond(O, H1)
    system.add_bond(O, H2)
    system.add_angle(H1, O, H2)

# System
system = System()

# Molecule 1
O1 = system.add_atom('O1', 'H20', '1', -0.834, np.array([3.084, 2.0, 0.0]))
H1 = system.add_atom('H1', 'H20', '1', 0.417, np.array([2.0, 1.0, -0.5]))
H2 = system.add_atom('H2', 'H20', '1', 0.417, np.array([4.0, 1.0, -0.5]))

system.add_bond(O1, H1)
system.add_bond(O1, H2)
system.add_angle(H1, O1, H2)

# Molecule 2
O2 = system.add_atom('O2', 'H20', '2', -0.834, np.array([3.0, 5.0, 0.0]))
H3 = system.add_atom('H3', 'H20', '2', 0.417, np.array([2.0, 6.0, 0.5]))
H4 = system.add_atom('H4', 'H20', '2', 0.417, np.array([4.0, 6.0, 0.5]))

system.add_bond(O2, H3)
system.add_bond(O2, H4)
system.add_angle(H3, O2, H4)

# Minimize & Analyze



if __name__ == '__main__':
    # Step 1: Bonded-only
    system.minimize(
        max_steps=5000,
        min_steps=100,
        stop_criteria='GRMS',
        threshold=0.01,
        bonded_only=True
    )

    # Step 2
    system.minimize(
        max_steps=5000,
        min_steps=100,
        stop_criteria='GRMS',
        threshold=0.01
    )

    system.plot_all()
    system.save('water.pdb')
