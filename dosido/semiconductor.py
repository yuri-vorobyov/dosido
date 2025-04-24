"""
Energy in eV. All energy levels are referenced with respect to the valence band edge.
Concentration in cm^-3.
Length in cm.
Temperature in K.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root, root_scalar

from dosido.constants import k_B, k_B_eV
from dosido.constants import m as m0
from dosido.dos_bands import BandTail, GaussianDefect, LocalisedStatesBand


class TemperatureDependentParams:

    def __init__(self, T: float, NC_300: float, NV_300: float, EA_300: float, ED_300: float, E0: float, K: float, QE: float, sU: float, X: float):
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
        # Linear scaling is applied to the defect band positions.
        self.T = 300.0
        Eg_300 = self.Eg
        self._alphaA = EA_300 / Eg_300
        self._alphaD = ED_300 / Eg_300
        # Temperature dependence of band tails follows temperature dependence of Urbach tail.
        self.T = 300.0
        gammaV_300 = 30e-3  # eV
        gammaC_300 = self.EU
        self._k_gamma = k_B_eV / self._sU
        self._b_gamma_C = gammaC_300 - self._k_gamma * 300.0
        self._b_gamma_V = gammaV_300 - self._k_gamma * 300.0
        # Finally, fix the value of temperature as provided by User.
        self.T = T

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
        EU = self.EU
        return self._b_gamma_C + self._k_gamma * self._T, self._b_gamma_V + self._k_gamma * self._T

    @property
    def EA(self):
        return self._alphaA * self.Eg

    @property
    def ED(self):
        return self._alphaD * self.Eg

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
        # Temperature dependencies.
        self.tdp = tdp
        # The very essence of the model --- DoS bands.
        self.vbt = vbt
        self.cbt = cbt
        self.acceptor = acceptor
        self.donor = donor
        # Update temperature-dependent parameters for the default temperature.
        self.T = 300.0

    @property
    def T(self):
        return self.tdp.T

    @T.setter
    def T(self, value):
        # set new value
        self.tdp.T = value
        # recalculate everything depending on it
        self.vbt.Eg = self.tdp.Eg
        self.cbt.Eg = self.tdp.Eg
        self.cbt.gamma, self.vbt.gamma = self.tdp.gammas
        self.acceptor.E0 = self.tdp.EA
        self.donor.E0 = self.tdp.ED

    def p(self, EFp):
        _, NV = self.tdp.NC_NV
        return NV * math.exp(-EFp / self.tdp.kT)

    def n(self, EFn):
        NC, _ = self.tdp.NC_NV
        return NC * math.exp((EFn - self.tdp.Eg) / self.tdp.kT)

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
            p_t = self.vbt.charge(lambda E: self.vbt.f_SRH(p, n, self.tdp.n1_p1(E)[1], self.tdp.n1_p1(E)[0]))
            n_t = self.cbt.charge(lambda E: self.cbt.f_SRH(p, n, self.tdp.n1_p1(E)[1], self.tdp.n1_p1(E)[0]))
            q_D = self.donor.charge(lambda E: self.donor.f_SRH(p, n, self.tdp.n1_p1(E)[1], self.tdp.n1_p1(E)[0]))
            q_A = self.acceptor.charge(lambda E: self.acceptor.f_SRH(p, n, self.tdp.n1_p1(E)[1], self.tdp.n1_p1(E)[0]))
            return p - n + p_t + n_t + q_D + q_A

        def f(x):
            p, n = x
            return generation_recombination_balance(p, n), charge_neutrality(p, n)

        # Initial guess is from equilibrium conditions, where the Fermi level is assumed between the donor and
        # acceptor bands.
        EF0 = self.solve_equilibrium()
        NC, NV = self.tdp.NC_NV
        kT = self.tdp.kT
        Eg = self.tdp.Eg
        p0 = NV * math.exp(-EF0 / kT)
        n0 = NC * math.exp(-Eg / kT) * math.exp(EF0 / kT)
        # print(f'initial guess: p0 = {p0:.2g} cm^-3 n0 = {n0:.2g} cm^-3')
        res = root(f, np.array([p0, n0]), tol=1e-12)
        if res.success:
            p, n = res.x
            # print(f'p = {p:.3g} cm^-3, n = {n:.3g} cm^-3')
            EFp = -math.log(p / NV) * kT
            EFn = math.log(n / (NC * math.exp(-Eg / kT))) * kT
            # print(f'EFp = {EFp:.3f} eV, EFn = {EFn:.3f} eV')
            return EFn, EFp
        else:
            print(res)
            raise RuntimeError(res.message)

    def _recombination_rate(self, band: LocalisedStatesBand, p: float, n: float):
        """
        Calculate the recombination rate due to the specific band of localised states in the bandgap.
        """
        prefactor = n * p - self.tdp.ni ** 2

        def integrand(E: float) -> float:
            num = band.Cn * band.Cp * band.density(E)
            n1, p1 = self.tdp.n1_p1(E)
            den = band.Cn * (n + n1) + band.Cp * (p + p1)
            return num / den

        res = quad(integrand, 0, self.tdp.Eg)
        return res[0] * prefactor


if __name__ == '__main__':
    TDP = TemperatureDependentParams(300, 3.9e21, 3.9e21, 0.941, 0.138, 214, 1.49, 9.13)
    print(TDP.ED)
