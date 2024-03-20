# Molecular Dynamics: Minimization

## Description

## Installation

1. Clone the repository

    ```sh
    git clone https://github.com/sebbeutler/md_minimize.git && cd md_minimize
    ```

2. Install dependencies

    ```sh
    pip install numpy matplotlib tqdm
    ```

## Usage

1. Import the module

    ```python
    from minimize import System
    ```

2. Create a system

    ```python
    system = System()
    ```

3. Add atoms and interactions

    ```python
    O1 = system.add_atom('O1', 'H20', '1', -0.834, np.array([3.084, 2.0]))
    H1 = system.add_atom('H1', 'H20', '1', 0.417, np.array([2.0, 1.0]))
    H2 = system.add_atom('H2', 'H20', '1', 0.417, np.array([4.0, 1.0]))

    system.add_bond(O1, H1)
    system.add_bond(O1, H2)
    system.add_angle(H1, O1, H2)
    ```

4. Perform the minimization

    ```python
    system.minimize(
        max_steps=5000,
        min_steps=100,
        stop_criteria='GRMS',
        threshold=0.01
    )
    ```

5. Analyze the results

    ```python
    system.plot_all()
    system.save('water.pdb')
    ```

## Run the example

System of 2 water molecules

    ```sh
    python water_system.py
    ```

## Bonus

Quick graphical playground to visualize the minimization in 2D.
Press the button minimize to toggle the minimization.
Atoms can me moved by drag&drop.

(Requirement)
    ```sh
    pip install pygame
    ```

Exection
    ```sh
    python minimize_gui.py
    ```

