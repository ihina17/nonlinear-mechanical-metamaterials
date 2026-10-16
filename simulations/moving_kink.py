import os
import numpy as np
import matplotlib.pyplot as plt

from src.model.parameters import omega0, c0


# Spatial domain
x = np.linspace(0, 20, 500)

# Initial kink center
x0 = 5.0

# Kink velocity
v = 0.5 * c0

# Time at which we evaluate the profile
t = 2.0

# Lorentz factor
gamma = 1 / np.sqrt(
    1 - (v / c0) ** 2
)


# Moving kink solution (1 velocity)
theta = 4 * np.arctan(
    np.exp(
        gamma
        * (omega0 / c0)
        * (x - v * t - x0)
    )
)


# Plot
plt.figure(figsize=(7, 5))

plt.plot(x, theta, linewidth=2)

plt.xlabel("Position x")
plt.ylabel(r"$\theta$")

plt.yticks(
    [0, np.pi, 2 * np.pi],
    [r"$0$", r"$\pi$", r"$2\pi$"]
)

plt.title(r"Moving Sine-Gordon Kink: $v = 0.5c_0$")

plt.grid()
plt.tight_layout()

plt.savefig(
    "figures/moving_kink.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#-----------------------


# kinks with different velocities

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

c0 = 2.0          # linear wave speed
ell = 0.5         # characteristic width parameter
x_center = 10.0   # keep all kinks centered here
x = np.linspace(0.0, 20.0, 1000)

# velocities as fractions of c0
velocity_fractions = [0.0, 0.5, 0.8, 0.95]


# ------------------------------------------------------------
# Moving kink profile
# ------------------------------------------------------------

def moving_kink(x, x_center, v, c0, ell):
    gamma = 1.0 / np.sqrt(1.0 - (v / c0) ** 2)
    return 4.0 * np.arctan(np.exp(gamma * (x - x_center) / ell))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for frac in velocity_fractions:
    v = frac * c0
    theta = moving_kink(x, x_center, v, c0, ell)

    if frac == 0.0:
        label = r"$v = 0$"
    else:
        label = rf"$v = {frac}c_0$"

    plt.plot(x, theta, linewidth=2.5, label=label)

plt.title("Moving Sine-Gordon Kink: Velocity Comparison", fontsize=18)
plt.xlabel("Position x", fontsize=14)
plt.ylabel(r"$\theta$", fontsize=18)

plt.yticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
plt.grid(True, alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/moving_kink_velocity_comparison.png", dpi=300)
plt.show()
