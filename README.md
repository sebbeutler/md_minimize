# Molecular Dynamics: Minimization

## Description


## Installation


`git clone https://github.com/sebbeutler/md_minimize.git && cd md_minimize`
`pip install numpy matplotlib tqdm`

## Usage

`python water_system.py`

```python
from mdsystem import *

# Base class
system = System()

# Coordinates
O = system.add_atom('O', np.array([4.0, 4.0]))
H1 = system.add_atom('H', np.array([5.0, 5.0]))
H2 = system.add_atom('H', np.array([3.5, 5.5]))

# Bounds
system.add_bound(O, H1)
system.add_bound(O, H2)
system.add_angle(H1, O, H2)

# Minimize
system.minimize()
# system.playground()

# Analyze
system.plot('energies', xlabel='steps', ylabel='E')
system.save('water.pdb')
```

## Reference

## Bonus
