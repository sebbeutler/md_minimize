import numpy as np
from minimize import System

# System
system = System()

def add_water(index: int, position: np.ndarray):
    O = system.add_atom(f'O', 'H20', str(index), -0.834, position)
    H1 = system.add_atom(f'H', 'H20', str(index), 0.417, position + np.array([-1.0, 1.0, 0.5]))
    H2 = system.add_atom(f'H', 'H20', str(index), 0.417, position + np.array([1.0, 1.0, 0.5]))

    system.add_bound(O, H1)
    system.add_bound(O, H2)
    system.add_angle(H1, O, H2)

mol = 1
for i in range(3):
    for j in range(3):
        for k in range(3):
            add_water(mol, np.array([i*3.0, j*3.0, k*3.0]))
            mol += 1


# Minimize & Analyze
# system.minimize(
#     max_steps=5000,
#     min_steps=100,
#     stop_criteria='GRMS',
#     threshold=0.01,
#     bounded_only=True
# )

system.minimize(
    max_steps=2000,
    min_steps=1,
    stop_criteria='GRMS',
    threshold=0.01
)

system.plot_all()
system.save('big_eau.pdb')


# from minimize_gui import SystemGui
# SystemGui(system).run()