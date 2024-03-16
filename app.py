from mdsystem import *

system = System()

O = system.add_atom('O', np.array([2.0, 2.0]))
H1 = system.add_atom('H', np.array([3.0, 3.0]))
H2 = system.add_atom('H', np.array([1.5, 2.5]))

system.add_bound(O, H1)
system.add_bound(O, H2)
system.add_angle(H1, O, H2)

O = system.add_atom('O', np.array([4.0, 4.0]))
H1 = system.add_atom('H', np.array([5.0, 5.0]))
H2 = system.add_atom('H', np.array([3.5, 5.5]))

system.add_bound(O, H1)
system.add_bound(O, H2)
system.add_angle(H1, O, H2)

# Analyze

system.minimize()
# system.playground()

system.plot('energies', xlabel='steps', ylabel='E')