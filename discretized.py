import time
import math
import numpy as np
from scipy.optimize import root_scalar

# Physical constants
k_B_eV = 8.617333262e-5  # eV K^-1

# Mathematical constants
C1 = 2 * math.sqrt(math.log(2.0) / math.pi)
C2 = 4 * math.log(2.0)

# Conditions
T = 300  # K
kT = k_B_eV * T  # eV

# Material parameters
E0 = 0.941  # eV
K = 0.138  # eV
QE = 214  # K
sU = 1.49
X = 9.13


def bandgap(temperature):
    return E0 - K / (np.exp(QE / temperature) - 1)  # eV


Eg = bandgap(T)

# Parameters for defect levels
epsA = 25e-3  # eV
epsD = 25e-3  # eV
Eg_300 = bandgap(300.0)
EA_300 = 0.57  # eV
ED_300 = 0.25  # eV
alphaA = EA_300 / Eg_300
alphaD = ED_300 / Eg_300
EA = alphaA * Eg
ED = alphaD * Eg
NA = 5e18
ND = 5e18


def Urbach_tail(temperature):
    return k_B_eV * QE / sU * ((1 + X) / 2 + 1 / (np.exp(QE / temperature) - 1))  # eV


EU = Urbach_tail(T)

# Parameters for band tails
g0C = 2e21
g0V = 2e21
EU_300 = Urbach_tail(300.0)
gammaV_300 = 30e-3  # eV
gammaC_300 = EU_300
k_gamma = k_B_eV / sU  # eV K^-1
b_gamma_C = gammaC_300 - k_gamma * 300.0
b_gamma_V = gammaV_300 - k_gamma * 300.0
gamma_C = b_gamma_C + k_gamma * T
gamma_V = b_gamma_V + k_gamma * T

# Parameters for extended states
NC_300 = 3.9e21  # cm^-3
NC = NC_300 * (T / 300) ** 1.5
NV_300 = 3.9e21  # cm^-3
NV = NV_300 * (T / 300) ** 1.5
ni = math.sqrt(NC * NV) * math.exp(-Eg / (2 * kT))  # cm^-3

# Computation parameters
grid_nodes_count = 1000
grid_node_width = Eg / grid_nodes_count  # eV
print(f'w = {grid_node_width * 1000:.2f} meV')
E = np.linspace(grid_node_width / 2.0, Eg - grid_node_width / 2.0, grid_nodes_count, dtype=np.float64)

# Discretization of DoS
cbt_N = g0C * np.exp((E - Eg) / gamma_C) * grid_node_width
vbt_N = g0V * np.exp(-E / gamma_V) * grid_node_width
acceptors_N = C1 * NA / epsA * np.exp(-C2 * ((E - EA) / epsA) ** 2) * grid_node_width
donors_N = C1 * ND / epsD * np.exp(-C2 * ((E - ED) / epsD) ** 2) * grid_node_width

# some vector constants could be precalculated to improve performance
exp_E_div_kT = np.exp(E / kT)


def eq_charge_neutrality(x):
    """
    Calculates the total charge density as a function of x:
    x = exp((EF - EV) / kT)
    """
    # mobile charge carriers
    p0 = NV / x
    n0 = -NC * math.exp(-Eg / kT) * x
    # Fermi-Dirac occupation function
    FD = 1 / (1 + exp_E_div_kT / x)
    # components of total charge
    pt = np.vecdot(vbt_N, 1 - FD)
    nt = -np.vecdot(cbt_N, FD)
    qA = -np.vecdot(acceptors_N, FD)
    qD = np.vecdot(donors_N, 1 - FD)
    return p0 + n0 + pt + nt + qD + qA


t0 = time.time()
sol = root_scalar(eq_charge_neutrality, bracket=[1, math.exp(Eg / kT)])
t1 = time.time()
print(f'`root_scalar` finished in {(t1 - t0) * 1000:.3f} ms')
if not sol.converged:
    raise RuntimeError(print(sol))
EF0 = math.log(sol.root) * kT
print(f'EF0 = {EF0:.3f} eV')
