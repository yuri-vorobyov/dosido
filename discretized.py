import time
import math
import numpy as np
from scipy.optimize import root_scalar, root

# Physical constants
k_B_eV = 8.617333262e-5  # eV K^-1

# Mathematical constants
C1 = 2 * math.sqrt(math.log(2.0) / math.pi)
C2 = 4 * math.log(2.0)

# Conditions
T = 300  # K
kT = k_B_eV * T  # eV
G = 1e18 * 1e4  # cm^-3 s^-1

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
EA_300 = 0.57  # eV
ED_300 = 0.25  # eV
CpA = 3.0e-11
CnA = 1.5e-12
CpD = 2.5e-12
CnD = 5.0e-11
Eg_300 = bandgap(300.0)
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
CpV = 5.0e-10
CnV = 5.0e-10
CpC = 5.0e-12
CnC = 5.0e-11
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
grid_nodes_count = 200
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
n1 = NC * exp_E_div_kT * math.exp(-Eg / kT)
p1 = NV / exp_E_div_kT


def equilibrium(x):
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
sol = root_scalar(equilibrium, bracket=[1, math.exp(Eg / kT)])
t1 = time.time()
print(f'`root_scalar` finished in {(t1 - t0) * 1000:.3f} ms')
if not sol.converged:
    raise RuntimeError(sol.message)
EF0 = math.log(sol.root) * kT
print(f'EF0 = {EF0:.3f} eV')


def solve_steady_state(G, guess):
    def fun(x):
        p, n = x

        def fs(Cp, Cn):
            r = Cn / Cp
            num_1 = r * n + p1
            num_2 = Cn
            den = r * (n + n1) + (p + p1)
            return num_1 / den, num_2 / den

        fV = fs(CpV, CnV)
        fC = fs(CpC, CnC)
        fA = fs(CpA, CnA)
        fD = fs(CpD, CnD)

        # for charge neutrality condition
        pt = np.vecdot(vbt_N, 1 - fV[0])
        nt = -np.vecdot(cbt_N, fC[0])
        qA = -np.vecdot(acceptors_N, fA[0])
        qD = np.vecdot(donors_N, 1 - fD[0])

        # for generation-recombination equality
        rV = np.vecdot(vbt_N, fV[1])
        rC = np.vecdot(cbt_N, fC[1])
        rA = np.vecdot(acceptors_N, fA[1])
        rD = np.vecdot(donors_N, fD[1])

        return p - n + pt + nt + qA + qD, G - (n * p - ni ** 2) * (rV + rC + rA + rD)

    EFp0, EFn0 = guess
    x0 = np.array([NV * math.exp(-EFp0 / kT), NC * math.exp((EFn0 - Eg) / kT)])
    sol = root(fun, x0)

    if sol.success:
        p, n = sol.x
        return True, -math.log(p / NV) * kT, math.log(n / (NC * math.exp(-Eg / kT))) * kT
    else:
        return False, None, None


EFp0, EFn0 = EF0, EF0
success, EFp, EFn = solve_steady_state(G, (EFp0, EFn0))
while not success:
    # looking for a value of generation rate which will provide us with a result (basically, for conditions closer to
    # equilibrium)
    g = G
    div = 2.0
    while not success:
        g /= div
        success, EFp, EFn = solve_steady_state(g, (EFp0, EFn0))
    # once it is found, more suitable initial guess is in our possession --- we could try it
    EFp0, EFn0 = EFp, EFn
    success, EFp, EFn = solve_steady_state(G, (EFp0, EFn0))
    # if `success == True` after that --- solution was found and the cycle will stop
    # if `success == False` ---  cycle will repeat itself again

print(f'EFp = {EFp:.3f} eV')
print(f'EFn = {EFn:.3f} eV')
