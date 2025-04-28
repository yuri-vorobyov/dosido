import numpy as np
import matplotlib.pyplot as plt

from dosido.dos_bands import BandTail, AllowedBand, GaussianDefect, ChargedState
from dosido.semiconductor import Semiconductor, TemperatureDependentParams

if __name__ == '__main__':
    tdp = TemperatureDependentParams(300, 3.9e21, 3.9e21, 0.57, 0.25, 0.941, 0.138, 214, 1.49, 9.13)
    vbt = BandTail(AllowedBand.VALENCE, 2e21, 32e-3, tdp.Eg, 5.0e-10, 5.0e-10)
    cbt = BandTail(AllowedBand.CONDUCTION, 2e21, 59e-3, tdp.Eg, 5.0e-12, 5.0e-11)
    acceptor = GaussianDefect(ChargedState.ACCEPTOR, 5e18, 25e-3, 0.57, 3.0e-11, 1.5e-12)
    donor = GaussianDefect(ChargedState.DONOR, 5e18, 25e-3, 0.25, 2.5e-12, 5.0e-11)
    gst = Semiconductor(tdp, vbt, cbt, acceptor, donor)

    G = 1e19 * 1e4  # cm^-3 s^-1

    EF0, EFp, EFn = [], [], []
    p0, n0, p, n = [], [], [], []

    temperature = np.linspace(200, 350, 20)

    _EFn, _EFp = 0.0, 0.0

    for T in temperature:
        gst.T = T
        print(f'\n{gst.T:.1f} K')

        # equilibrium state
        EF0.append(gst.solve_equilibrium())
        print(f'EF0 = {EF0[-1]:.3f} eV')
        p0.append(gst.p(EF0[-1]))
        print(f'p0 = {p0[-1]:.3g} cm^-3')
        n0.append(gst.n(EF0[-1]))
        print(f'n0 = {n0[-1]:.3g} cm^-3')

        # for the very first temperature only equilibrium Fermi level is available
        if T == temperature[0]:
            _EFn, _EFp = EF0[-1], EF0[-1]

        # steady state
        g = G
        k = 0
        while True:
            try:
                _EFn, _EFp = gst.solve_steady_state(g, (_EFp, _EFn))
                break
            except RuntimeError:
                g /= 10
                k += 1
        # If the solution was obtained by decreasing the generation rate --- move back up using previous result as
        # the initial guess.
        for i in range(k):
            g *= 10
            _EFn, _EFp = gst.solve_steady_state(g, (_EFp, _EFn))
        EFp.append(_EFp)
        print(f'EFp = {EFp[-1]:.3f} eV')
        EFn.append(_EFn)
        print(f'EFn = {EFn[-1]:.3f} eV')
        p.append(gst.p(EFp[-1]))
        print(f'p = {p[-1]:.3g} cm^-3')
        n.append(gst.n(EFn[-1]))
        print(f'n = {n[-1]:.3g} cm^-3')

    plt.plot(temperature, np.array(p) - np.array(p0))
    plt.yscale('log')
    plt.show()
