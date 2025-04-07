"""
Energy in eV. All energy levels are referenced with respect to the valence band edge.
Concentration in cm^-3.
Length in cm.
Temperature in K.
"""
import math
from typing import List

import numpy as np
import scipy.optimize
from scipy.integrate import quad
from scipy.optimize import root, root_scalar
from dosido.constants import k_B, k_B_eV
from dosido.constants import m as m0
from dosido.dos_bands import BandTail, AllowedBand, ChargedState, GaussianDefect, LocalisedStatesBand


class Semiconductor:

    def __init__(self, vbt: BandTail, cbt: BandTail, acceptor: GaussianDefect, donor: GaussianDefect):
        """

        """
        # default parameters correspond to the room temperature
        self._T = 300
        self._Eg = 0.8
        self._NC_300 = 3.9e21
        self._NV_300 = 3.9e21
        self._kT = k_B_eV * self._T
        self.vbt = vbt
        self.cbt = cbt
        self.acceptor = acceptor
        self.donor = donor

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value
        self._kT = k_B_eV * self._T

    @property
    def Eg(self):
        return self._Eg

    @Eg.setter
    def Eg(self, value):
        self._Eg = value

    @property
    def NC(self):
        return self._NC_300 / 300 ** 1.5 * self._T ** 1.5

    @property
    def NV(self):
        return self._NV_300 / 300 ** 1.5 * self._T ** 1.5

    @property
    def ni(self):
        return math.sqrt(self.NC * self.NV) * math.exp(-self._Eg / (2 * self._kT))

    def n1(self, energy: float):
        return self.NC * math.exp((energy - self._Eg) / self._kT)

    def p1(self, energy: float):
        return self.NV * math.exp(-energy / self._kT)

    @property
    def vth(self):
        return math.sqrt(3 * k_B * self._T / m0) * 100.0

    def solve_equilibrium(self):
        """
        Calculate the equilibrium position of Fermi level.
        """
        def fun(x):
            """
            Calculates the total charge density as a function of x:
            x = exp((EF - EV) / kT)
            """
            p0 = self.NV / x
            n0 = -self.NC * math.exp(-self.Eg / self._kT) * x
            pt = self.vbt.charge(lambda E: 1 / (1 + math.exp(E / self._kT) / x))
            nt = self.cbt.charge(lambda E: 1 / (1 + math.exp(E / self._kT) / x))
            qD = self.donor.charge(lambda E: 1 / (1 + math.exp(E / self._kT) / x))
            qA = self.acceptor.charge(lambda E: 1 / (1 + math.exp(E / self._kT) / x))
            return p0 + n0 + pt + nt + qD + qA

        sol = root_scalar(fun, bracket=[1, math.exp(self.Eg / self._kT)])
        if not sol.converged:
            raise RuntimeError(print(sol))
        EF0 = math.log(sol.root) * self._kT
        return EF0

    def solve_steady_state(self, G: float):
        """

        """

        def generation_recombination_balance(p, n):
            R = self._recombination_rate(self.vbt, p, n) + \
                self._recombination_rate(self.donor, p, n) + \
                self._recombination_rate(self.acceptor, p, n) + \
                self._recombination_rate(self.vbt, p, n)
            return G - R

        def charge_neutrality(p, n):
            p_t = self.vbt.charge(lambda E: self.vbt.f_SRH(p, n, self.p1(E), self.n1(E)))
            n_t = self.cbt.charge(lambda E: self.cbt.f_SRH(p, n, self.p1(E), self.n1(E)))
            q_D = self.donor.charge(lambda E: self.donor.f_SRH(p, n, self.p1(E), self.n1(E)))
            q_A = self.acceptor.charge(lambda E: self.acceptor.f_SRH(p, n, self.p1(E), self.n1(E)))
            return p - n + p_t + n_t + q_D + q_A

        def f(x):
            p, n = x
            return generation_recombination_balance(p, n), charge_neutrality(p, n)

        # Initial guess is from equilibrium conditions, where the Fermi level is assumed between the donor and
        # acceptor bands.
        EF0 = 0.4
        p0 = self.NV * math.exp(-EF0 / self._kT)
        n0 = self.NC * math.exp(-self._Eg / self._kT) * math.exp(EF0 / self._kT)
        print(f'initial guess: p0 = {p0:.2g} cm^-3 n0 = {n0:.2g} cm^-3')
        res = root(f, np.array([p0, n0]), tol=1e-12)
        print(res)
        p, n = res.x
        EFp = -math.log(p / self.NV) * self._kT
        EFn = math.log(n / (self.NC * math.exp(-self._Eg / self._kT))) * self._kT
        print(f'EFp = {EFp:.3f} eV, EFn = {EFn:.3f} eV')

    def _recombination_rate(self, band: LocalisedStatesBand, p: float, n: float):
        """
        Calculate the recombination rate due to the specific band of localised states in the bandgap.
        """
        prefactor = n * p - self.ni ** 2

        def integrand(E: float) -> float:
            num = band.Cn * band.Cp * band.density(E)
            den = band.Cn * (n + self.n1(E)) + band.Cp * (p + self.p1(E))
            return num / den

        res = quad(integrand, 0, self.Eg)
        return res[0] * prefactor


if __name__ == '__main__':
    gst = Semiconductor()
    print(gst.vbt.N)
    print(gst.vbt.occupied(lambda e: 1.0))
    print(gst.cbt.N)
    print(gst.cbt.occupied(lambda e: 1.0))
    print(gst.vth)
    print(f'EF0 = {gst.solve_equilibrium():.3f} eV')
    gst.solve_steady_state(5e18 * 1e0)  # cm^-3 s^-1
