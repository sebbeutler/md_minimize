import numpy as np
from mdsystem import System as System1
from mdsystem import function_times
from mdsystem_copy import System as System2
import time
import random

random.seed(42)
np.random.seed(42)

# System
def md_time(system):
    start = time.time()
    # Water 1
    O = system.add_atom('O', np.array([2.0, 2.0, 0.0]))
    H1 = system.add_atom('H', np.array([3.0, 3.0, 0.0]))
    H2 = system.add_atom('H', np.array([1.5, 2.5, 0.0]))

    system.add_bound(O, H1)
    system.add_bound(O, H2)
    system.add_angle(H1, O, H2)

    # Water 2
    O = system.add_atom('O', np.array([4.0, 4.0, 0.0]))
    H1 = system.add_atom('H', np.array([5.0, 5.0, 0.0]))
    H2 = system.add_atom('H', np.array([3.5, 5.5, 0.0]))

    system.add_bound(O, H1)
    system.add_bound(O, H2)
    system.add_angle(H1, O, H2)

    for i in range(3):
        system.add_atom(random.choice(['H', 'O']), np.random.uniform(low=-10, high=10, size=3))

    # Minimize & Analyze
    system.minimize(max_steps=500, threshold=0.000000000001)
    # system.plot_all()
    print(system.step)
    print('Time: ', time.time() - start, 's')


md_time(System1())
for func_name, total_time in function_times.items():
    print(f"Function '{func_name}' total execution time: {total_time} seconds")
# md_time(System2())

# Playground
# from mdsystem_gui import SystemGui

# gui = SystemGui(system)
# gui.run()
