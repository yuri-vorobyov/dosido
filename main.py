from dosido.dos_bands import BandTail, AllowedBand, GaussianDefect, ChargedState
from dosido.semiconductor import Semiconductor, TemperatureDependentParams

if __name__ == '__main__':
    tdp = TemperatureDependentParams(300, 3.9e21, 3.9e21, 0.57, 0.25, 0.941, 0.138, 214, 1.49, 9.13)
    vbt = BandTail(AllowedBand.VALENCE, 2e21, 32e-3, tdp.Eg, 5.0e-10, 5.0e-10)
    cbt = BandTail(AllowedBand.CONDUCTION, 2e21, 59e-3, tdp.Eg, 5.0e-12, 5.0e-11)
    acceptor = GaussianDefect(ChargedState.ACCEPTOR, 5e18, 25e-3, 0.57, 3.0e-11, 1.5e-12)
    donor = GaussianDefect(ChargedState.DONOR, 5e18, 25e-3, 0.25, 2.5e-12, 5.0e-11)
    gst = Semiconductor(tdp, vbt, cbt, acceptor, donor)

    gst.T = 200.0

    # equilibrium state
    EF0 = gst.solve_equilibrium()
    print(f'EF0 = {EF0:.3f} eV')
    print(f'p0 = {gst.p(EF0):.3g} cm^-3')
    print(f'n0 = {gst.n(EF0):.3g} cm^-3')

    # steady state
    EFn, EFp = gst.solve_steady_state(1e18 * 1e4)  # cm^-3 s^-1
    print(f'EFp = {EFp:.3f} eV')
    print(f'EFn = {EFn:.3f} eV')
    print(f'p = {gst.p(EFp):.3g} cm^-3')
    print(f'n = {gst.n(EFn):.3g} cm^-3')
