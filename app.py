from mdsystem import *

system = System()

O = system.add_atom('O', np.array([2.0, 2.0, 0.0]))
# H1 = system.add_atom('H', np.array([3.0, 3.0, 0.0]))
# H2 = system.add_atom('H', np.array([1.5, 2.5, 0.0]))

# system.add_bound(O, H1)
# system.add_bound(O, H2)
# system.add_angle(H1, O, H2)

O = system.add_atom('O', np.array([4.0, 4.0, 0.0]))
# H1 = system.add_atom('H', np.array([5.0, 5.0, 0.0]))
# H2 = system.add_atom('H', np.array([3.5, 5.5, 0.0]))

# system.add_bound(O, H1)
# system.add_bound(O, H2)
# system.add_angle(H1, O, H2)

# Analyze

# system.minimize(max_steps=1000)

from mdsystem_gui import SystemGui

gui = SystemGui(system)
gui.run()

# Plot

import matplotlib.pyplot as plt

steps = np.arange(system.step)

def set_size(w: int, h: int):
    """ w, h: width, height in inches """
    ax = plt.gca()
    l = ax.figure.subplotpars.left  # type: ignore
    r = ax.figure.subplotpars.right # type: ignore
    t = ax.figure.subplotpars.top   # type: ignore
    b = ax.figure.subplotpars.bottom    # type: ignore
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)   # type: ignore

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6))= plt.subplots(2, 3)

fig.set_dpi(150.0)
fig.tight_layout()
set_size(9, 5)

ax1.set_ylabel('E')
ax1.set_title('Energie')
# ax1.set_ylim([0,2])
ax1.grid(True)
ax1.plot(steps, system.metrics['energies'])

ax2.set_ylabel('gradient')
ax2.set_title('Max Gradient')
ax2.grid(True)
ax2.plot(steps, system.metrics['max_gradients'])

ax3.set_ylabel('GRMS')
ax3.set_title('GRMS')
ax3.set_ylim([0,2])
ax3.grid(True)
ax3.plot(steps, system.metrics['GRMS'])

ax4.set_ylabel('E')
ax4.set_title('Energie difference')
ax4.grid(True)
ax4.plot(steps, system.metrics['energie_diff'])

ax5.set_ylabel('RMSD')
ax5.set_title('RMSD')
ax5.grid(True)
ax5.plot(steps, system.metrics['RMSD'])

plt.show()