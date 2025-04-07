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


class TemperatureDependentParams:

    def __init__(self, T: float, NC_300: float, NV_300: float, E0: float, K: float, QE: float, sU: float, X: float):
        self._T = T
        self._kT = k_B_eV * T
        # Temperature dependence of effective density of states.
        self._NC_300 = NC_300
        self._NV_300 = NV_300
        # Temperature dependence of the bandgap is defined using following parameters
        self._E0 = E0  # eV
        self._K = K  # eV
        self._QE = QE  # K
        # To define Urbach edge, additional parameters are needed.
        self._sU = sU
        self._X = X

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value
        self._kT = k_B_eV * value

    @property
    def kT(self):
        return self._kT

    @property
    def vth(self):
        return math.sqrt(3 * k_B * self._T / m0) * 100.0

    @property
    def NC_NV(self):
        den = 300 ** 1.5
        num = self._T ** 1.5
        return self._NC_300 / den * num, self._NV_300 / den * num

    @property
    def Eg(self):
        return self._E0 - self._K / (math.exp(self._QE / self._T) - 1)

    @property
    def EU(self):
        return k_B_eV * self._QE / self._sU * ((1 + self._X) / 2 + 1 / (math.exp(self._QE / self._T) - 1))

    @property
    def gammas(self):
        gammaV = 30e-3  # eV
        n = 4.0
        gammaC = (self.EU ** n - gammaV ** n) ** (1 / n)
        return gammaC, gammaV

    @property
    def ni(self):
        NC, NV = self.NC_NV
        return math.sqrt(NC * NV) * math.exp(-self.Eg / (2 * self._kT))

    def n1_p1(self, energy: float):
        NC, NV = self.NC_NV
        return NC * math.exp((energy - self.Eg) / self._kT), NV * math.exp(-energy / self._kT)


class Semiconductor:

    def __init__(self, tdp: TemperatureDependentParams, vbt: BandTail, cbt: BandTail, acceptor: GaussianDefect, donor: GaussianDefect):
        """

        """
        # Semiconductor instance always has some definite temperature.
        self.tdp = tdp
        self.tdp.T = 300.0
        # The very essence of the model --- DoS bands.
        self.vbt = vbt
        self.cbt = cbt
        self.cbt.gamma, self.vbt.gamma = self.tdp.gammas
        self.acceptor = acceptor
        self.donor = donor

    def solve_equilibrium(self):
        """
        Calculate the equilibrium position of Fermi level.
        """
        kT = self.tdp.kT
        NC, NV = self.tdp.NC_NV

        def fun(x):
            """
            Calculates the total charge density as a function of x:
            x = exp((EF - EV) / kT)
            """
            p0 = NV / x
            n0 = -NC * math.exp(-self.tdp.Eg / kT) * x
            pt = self.vbt.charge(lambda E: 1 / (1 + math.exp(E / kT) / x))
            nt = self.cbt.charge(lambda E: 1 / (1 + math.exp(E / kT) / x))
            qD = self.donor.charge(lambda E: 1 / (1 + math.exp(E / kT) / x))
            qA = self.acceptor.charge(lambda E: 1 / (1 + math.exp(E / kT) / x))
            return p0 + n0 + pt + nt + qD + qA

        sol = root_scalar(fun, bracket=[1, math.exp(self.tdp.Eg / kT)])
        if not sol.converged:
            raise RuntimeError(print(sol))
        EF0 = math.log(sol.root) * kT
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

