import math
from typing import Callable
import abc
from enum import Enum

import numpy as np
from scipy.integrate import quad


class ChargedState(Enum):
    # Acceptors are neutral when empty and negatively charged when occupied by an electron. Their ionisation is
    # accomplished by an electron capture.
    ACCEPTOR = 'A'
    # Donors are neutral when occupied by an electron and positively charged when empty. Their ionisation is more like
    # a hole capture.
    DONOR = 'D'


class AllowedBand(Enum):
    VALENCE = 'V'
    CONDUCTION = 'C'


class LocalisedStatesBand(object, metaclass=abc.ABCMeta):
    """
    Base class for various types of bands in band gap.
    """

    def __init__(self, kind: ChargedState, Cp: float, Cn: float):
        """
        :param kind: The way states change their charge changing their occupancy.
        :param Cp: Capture coefficient for holes.
        :param Cn: Capture coefficient for electrons.
        """
        self._kind = kind
        self._Cp = Cp
        self._Cn = Cn

    @property
    def Cp(self):
        return self._Cp

    @Cp.setter
    def Cp(self, value):
        self._Cp = value

    @property
    def Cn(self):
        return self._Cn

    @Cn.setter
    def Cn(self, value):
        self._Cn = value

    @property
    @abc.abstractmethod
    def N(self):
        """Should return the total number of states in the band."""

    @abc.abstractmethod
    def density(self, energy: float) -> float:
        """Calculate the energy distribution of allowed states density."""

    @abc.abstractmethod
    def occupied(self, f: Callable[[float], float]) -> float:
        """Calculate the number of state in the band which are occupied according to occupation function `f`."""

    @abc.abstractmethod
    def charge(self, f: Callable[[float], float]) -> float:
        """Calculate the total charge density in the band according to occupation function `f`."""

    def f_SRH(self, p: float, n: float, p1: float, n1: float) -> float:
        """Calculate the SRH occupancy function for the states of this particular band."""
        r = self._Cn / self._Cp
        num = r * n + p1
        den = r * (n + n1) + (p + p1)
        return num / den


class GaussianDefect(LocalisedStatesBand):
    C1 = 2 * math.sqrt(math.log(2) / math.pi)
    C2 = 4 * math.log(2)

    def __init__(self, kind: ChargedState, N: float, eps: float, E0: float, Cp: float, Cn: float):
        """
        :param N: Total number of states in the Gaussian band.
        :param eps: Full width at half-maximum of the Gaussian band.
        :param E0: Position of the Gaussian band.
        """
        super().__init__(kind, Cp, Cn)
        self._N = N
        self._eps = eps
        self._E0 = E0

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value

    @property
    def E0(self):
        return self._E0

    @E0.setter
    def E0(self, value):
        self._E0 = value

    def density(self, energy: float) -> float:
        return self.C1 * self._N / self._eps * np.exp(-self.C2 * ((energy - self._E0) / self._eps) ** 2)

    def occupied(self, f: Callable[[float], float]) -> float:
        C = 2
        e0, e1 = self._E0 - C * self._eps, self._E0 + C * self._eps
        res = quad(lambda e: self.density(e) * f(e), e0, e1)
        return res[0]

    def charge(self, f: Callable[[float], float]) -> float:
        if self._kind == ChargedState.ACCEPTOR:
            # acceptor is negative when occupied by an electron
            return -1 * self.occupied(f)
        elif self._kind == ChargedState.DONOR:
            # donor is positive when empty
            return +1 * (self._N - self.occupied(f))


class BandTail(LocalisedStatesBand):

    def __init__(self, kind: AllowedBand, g0: float, gamma: float, Eg: float, Cp: float, Cn: float):
        if kind == AllowedBand.VALENCE:
            super().__init__(ChargedState.DONOR, Cp, Cn)
        elif kind == AllowedBand.CONDUCTION:
            super().__init__(ChargedState.ACCEPTOR, Cp, Cn)
        self._g0 = g0
        self._gamma = gamma
        self._kind = kind
        self._Eg = Eg

    @property
    def g0(self):
        return self._g0

    @g0.setter
    def g0(self, value):
        self._g0 = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def Eg(self):
        return self._Eg

    @Eg.setter
    def Eg(self, value):
        self._Eg = value

    @property
    def N(self):
        return self._g0 * self._gamma

    def density(self, energy: float):
        if self._kind == AllowedBand.VALENCE:
            dE = 0 - energy
        elif self._kind == AllowedBand.CONDUCTION:
            dE = energy - self._Eg
        else:
            raise ValueError('check `kind` property')
        return self._g0 * np.exp(dE / self._gamma)

    def occupied(self, f: Callable[[float], float]):
        res = quad(lambda e: self.density(e) * f(e), 0, self._Eg)
        return res[0]

    def charge(self, f: Callable[[float], float]):
        if self._kind == AllowedBand.VALENCE:
            # VBT is donor-like, i.e. positive when empty
            return +1 * (self.N - self.occupied(f))
        elif self._kind == AllowedBand.CONDUCTION:
            # CBT is acceptor-like, i.e. negative when occupied
            return -1 * self.occupied(f)
