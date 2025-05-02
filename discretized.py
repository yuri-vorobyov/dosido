import time
import math
import numpy as np
from scipy.optimize import root_scalar, root
import matplotlib.pyplot as plt

# Physical constants
k_B_eV = 8.617333262e-5  # eV K^-1

# Mathematical constants
C1 = 2 * math.sqrt(math.log(2.0) / math.pi)
C2 = 4 * math.log(2.0)

material = dict(
    # Temperature dependence of Eg and EU
    Eg_300 = 0.808,  # eV
    EU_300 = 0.0746, # eV
    K = 0.138,  # eV
    QE = 214,  # K
    sU = 1.49,
    # Extended states
    NC_300 = 3.9e21,  # cm^-3
    NV_300 = 3.9e21,  # cm^-3
    # Band tails
    g0C = 2e21,
    g0V = 2e21,
    CpV = 5.0e-10,
    CnV = 5.0e-10,
    CpC = 5.0e-12,
    CnC = 5.0e-11,
    gammaV_300 = 30e-3,  # eV
    gammaC_300 = 0.0746,  # eV
    # Defect levels
    epsA = 25e-3,  # eV
    epsD = 25e-3,  # eV
    EA_300 = 0.57,  # eV
    ED_300 = 0.25,  # eV
    CpA = 3.0e-11,
    CnA = 1.5e-12,
    CpD = 2.5e-12,
    CnD = 5.0e-11,
    NA = 5e18,
    ND = 5e18
)


def solve(material, T, G, guess=None):
    kT = k_B_eV * T  # eV

    # calculated parameters of the material (temperature independent)
    alphaA = material['EA_300'] / material['Eg_300']
    alphaD = material['ED_300'] / material['Eg_300']
    k_gamma = k_B_eV / material['sU']  # eV K^-1
    b_gamma_C = material['gammaC_300'] - k_gamma * 300.0
    b_gamma_V = material['gammaV_300'] - k_gamma * 300.0

    # temperature dependent material parameters
    Eg = material['Eg_300'] - material['K'] / material['QE'] * (T - 300.0)  # eV
    EA = alphaA * Eg
    ED = alphaD * Eg
    EU = material['EU_300'] + k_B_eV / material['sU'] * (T - 300.0)  # eV
    gamma_C = b_gamma_C + k_gamma * T
    gamma_V = b_gamma_V + k_gamma * T
    NC = material['NC_300'] * (T / 300) ** 1.5
    NV = material['NV_300'] * (T / 300) ** 1.5
    ni = math.sqrt(NC * NV) * math.exp(-Eg / (2 * kT))  # cm^-3

    # Computation parameters
    grid_nodes_count = 200
    grid_node_width = Eg / grid_nodes_count  # eV
    E = np.linspace(grid_node_width / 2.0, Eg - grid_node_width / 2.0, grid_nodes_count, dtype=np.float64)

    # Discretization of DoS
    cbt_N = material['g0C'] * np.exp((E - Eg) / gamma_C) * grid_node_width
    vbt_N = material['g0V'] * np.exp(-E / gamma_V) * grid_node_width
    acceptors_N = C1 * material['NA'] / material['epsA'] * np.exp(-C2 * ((E - EA) / material['epsA']) ** 2) * grid_node_width
    donors_N = C1 * material['ND'] / material['epsD'] * np.exp(-C2 * ((E - ED) / material['epsD']) ** 2) * grid_node_width

    # some vector constants could be precalculated to improve performance
    exp_E_div_kT = np.exp(E / kT)
    n1 = NC * exp_E_div_kT * math.exp(-Eg / kT)
    p1 = NV / exp_E_div_kT

    # solve for equilibrium conditions
    def fun(x):
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

    sol = root_scalar(fun, bracket=[1, math.exp(Eg / kT)])
    if not sol.converged:
        raise RuntimeError(sol.message)
    EF0 = math.log(sol.root) * kT

    def solve_steady_state(G, guess):
        def fun(x):
            p, n = x

            def fs(Cp, Cn):
                r = Cn / Cp
                num_1 = r * n + p1
                num_2 = Cn
                den = r * (n + n1) + (p + p1)
                return num_1 / den, num_2 / den

            fV = fs(material['CpV'], material['CnV'])
            fC = fs(material['CpC'], material['CnC'])
            fA = fs(material['CpA'], material['CnA'])
            fD = fs(material['CpD'], material['CnD'])

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

        x0 = np.array([NV * math.exp(-guess[0] / kT), NC * math.exp((guess[1] - Eg) / kT)])
        sol = root(fun, x0)

        if sol.success:
            p, n = sol.x
            return True, -math.log(p / NV) * kT, math.log(n / (NC * math.exp(-Eg / kT))) * kT
        else:
            return False, None, None

    if not guess:
        EFp0, EFn0 = EF0, EF0
    else:
        EFp0, EFn0 = guess
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
    
    return EF0, EFp, EFn


# temperature dependence
temperature = np.linspace(200, 350, 100)
EF0 = np.zeros_like(temperature)
EFp = np.zeros_like(temperature)
EFn = np.zeros_like(temperature)

t0 = time.time()

EF0[0], EFp[0], EFn[0] = solve(material, temperature[0], 1.0e18 * 1.0e4)
for i in range(1, len(temperature)):
    T = temperature[i]
    EF0[i], EFp[i], EFn[i] = solve(material, T, 1.0e18 * 1.0e4, (EFp[i-1], EFn[i-1]))

t1 = time.time()
print(f'solved in {(t1 - t0) * 1000:.3f} ms')

plt.plot(temperature, EFp)
plt.show()
