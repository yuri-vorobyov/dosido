from dosido.dos_bands import BandTail, AllowedBand, GaussianDefect, ChargedState
from dosido.semiconductor import Semiconductor


if __name__ == '__main__':
    Eg = 0.8
    vbt = BandTail(AllowedBand.VALENCE, 2e21, 32e-3, Eg, 5.0e-10, 5.0e-10)
    cbt = BandTail(AllowedBand.CONDUCTION, 2e21, 59e-3, Eg, 5.0e-12, 5.0e-11)
    acceptor = GaussianDefect(ChargedState.ACCEPTOR, 5e18, 25e-3, 0.57, 3.0e-11, 1.5e-12)
    donor = GaussianDefect(ChargedState.DONOR, 5e18, 25e-3, 0.25, 2.5e-12, 5.0e-11)
    gst = Semiconductor(vbt, cbt, acceptor, donor)
    print(f'EF0 = {gst.solve_equilibrium():.3f} eV')
    gst.solve_steady_state(5e18 * 1e4)  # cm^-3 s^-1
