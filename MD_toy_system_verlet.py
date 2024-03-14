import math

DT = 0.001        # time step
DT2 = DT*DT       # time step squared
NSTEPS = 10000 # total nb of steps for the MD
DELTA = 0.001     # small increment to calc Epot derivative numerically
XINIT = -2        # initial x coor
TEMPINIT = 5      # initial temperature
L0 = 0.9572
KB = 450.0

def calc_Epot(x):
    Epot = KB*(x - L0)**2
    return Epot

#calculate force
def calc_f(x):
    f = - (calc_Epot(x+DELTA) - calc_Epot(x-DELTA)) / (2 * DELTA)
    return f

def calc_Ekin_T(v):
    Ekin = .5 * v**2 # Ekin = .5mv**2 (fictitious mass set to 1)
    T = 2 * Ekin # Ekin = .5 kT * Ndof <=> Ekin = .5 T <=> T = 2 Ekin
    return Ekin, T

x0 = XINIT
v0 = TEMPINIT**.5
a0 = calc_f(x0)

x1 = x0 + (v0 * DT) + (.5 * a0 * DT2)

x_prev = x0
x = x1

epots = []

for i in range(1, NSTEPS + 1):
    # Calculate next coor (xnew) using Verlet
    x_new = (2 * x) - x_prev + (DT2 * calc_f(x))
    
    # calculate current velocity, Ekin and T
    v = (x_new - x_prev) / (2 * DT)
    
    # calc Energies and T
    Epot = calc_Epot(x)
    Ekin, T = calc_Ekin_T(v)
    Etot = Ekin + Epot
    
    epots.append(Epot)
    
    # Update coor for next iteration.
    x_prev = x
    x = x_new

import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('ts')
plt.ylabel('E')
plt.title('Energie')
plt.plot(range(len(epots)), epots, label="E_potential")
plt.legend(loc="upper left")