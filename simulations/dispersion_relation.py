import numpy as np
import matplotlib.pyplot as plt

from src.model.parameters import m, l, g, kappa, a, omega0

# First Brillouin zone

k = np.linspace(-np.pi/a, np.pi/a, 500)

# Discrete dispersion relation

omega = np.sqrt(
    omega0**2 + (4 + kappa/(m* l**2)) * np.sin(k* a/2)**2
)

#Plot

plt.figure(figsize = (7, 5))

# plot the forbidden region
plt.fill_between(k, 0, omega0, alpha = 0.3, label = "Forbidden region")
#plot the dispersion curve
plt.plot(k, omega, label = "Dispersion relation")
plt.axhline(omega0, linestyle = "--", linewidth = 1, label = r"$\omega_0$")
plt.xlabel(r"Wavenumber (q)")
plt.ylabel(r"Angular frequency ($\omega$)")
plt.title("Dispersion relation of the discrete pendulum lattice")
plt.grid()
plt.tight_layout()

plt.savefig(
    "figures/dispersion_relation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()