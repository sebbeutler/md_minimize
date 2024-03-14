from system import *

system = System()

# Atoms
O = system.add_atom('O', np.array([2.0, 2.0]))
H1 = system.add_atom('H', np.array([3.0, 3.0]))
H2 = system.add_atom('H', np.array([1.5, 2.5]))


C1 = system.add_atom('C', np.array([4.0, 4.0]))
C2 = system.add_atom('C', np.array([5.0, 4.0]))
C3 = system.add_atom('C', np.array([4.5, 5.0]))

# Interactions
system.add_bound(O, H1)
system.add_bound(O, H2)
system.add_angle(H1, O, H2)

system.add_bound(C1, C2)
system.add_bound(C1, C3)
system.add_bound(C2, C3)

# Analyzes
system.run()