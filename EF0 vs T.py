import matplotlib.pyplot as plt
import numpy as np

from dosido.dos_bands import BandTail, AllowedBand, GaussianDefect, ChargedState
from dosido.semiconductor import TemperatureDependentParams, Semiconductor

plt.style.use('style.mplstyle')
plt.rcParams['savefig.directory'] = '.'
COLORS = [item['color'] for item in plt.rcParams['axes.prop_cycle'].__dict__['_left']]

# creating figure
fig, ax_T = plt.subplots(1, 1)
fig.canvas.manager.set_window_title('figure')

# configuring axes
ax_T.set_xlabel(r'Temperature (°C)')
ax_T.set_ylabel(r'Fermi level (eV)')

tdp = TemperatureDependentParams(300, 3.9e21, 3.9e21, 0.941, 0.138, 214, 1.49, 9.13)
vbt = BandTail(AllowedBand.VALENCE, 2e18, 32e-3, tdp.Eg, 5.0e-10, 5.0e-10)
cbt = BandTail(AllowedBand.CONDUCTION, 2e18, 59e-3, tdp.Eg, 5.0e-12, 5.0e-11)
acceptor = GaussianDefect(ChargedState.ACCEPTOR, 5e17, 25e-3, 0.55, 3.0e-11, 1.5e-12)
donor = GaussianDefect(ChargedState.DONOR, 5e17, 25e-3, 0.24, 2.5e-12, 5.0e-11)

gst = Semiconductor(tdp, vbt, cbt, acceptor, donor)
temperatures = np.linspace(-70, 60, 50) + 273.15  # K

EF0s = []
for temperature in temperatures:
    gst.T = temperature
    EF0s.append(gst.solve_equilibrium())

ax_T.plot(temperatures, EF0s, alpha=0.7)

plt.show()
