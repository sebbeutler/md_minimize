import numpy as np
from mdsystem import System

# System
system = System()


# Water molecule 1
O1 = system.add_atom('O1', 'H20', '1', -0.834, np.array([3.084, 2.0, 0.0]))
H1 = system.add_atom('H1', 'H20', '1', 0.417, np.array([2.0, 1.0, -0.5]))
H2 = system.add_atom('H2', 'H20', '1', 0.417, np.array([4.0, 1.0, -0.5]))

system.add_bound(O1, H1)
system.add_bound(O1, H2)
system.add_angle(H1, O1, H2)

# Water molecule 2
O2 = system.add_atom('O2', 'H20', '2', -0.834, np.array([3.0, 5.0, 0.0]))
H3 = system.add_atom('H3', 'H20', '2', 0.417, np.array([2.0, 6.0, 0.5]))
H4 = system.add_atom('H4', 'H20', '2', 0.417, np.array([4.0, 6.0, 0.5]))

system.add_bound(O2, H3)
system.add_bound(O2, H4)
system.add_angle(H3, O2, H4)


# Minimize & Analyze
system.minimize(
    max_steps=5000,
    min_steps=100,
    stop_criteria='GRMS',
    threshold=0.01,
    bounded_only=True
)

system.minimize(
    max_steps=5000,
    min_steps=100,
    stop_criteria='GRMS',
    threshold=0.01
)

system.plot_all()
system.save('aout.pdb')
