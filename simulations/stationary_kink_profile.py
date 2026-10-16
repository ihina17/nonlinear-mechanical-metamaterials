import numpy as np
import matplotlib.pyplot as plt

from src.model.parameters import omega0, c0


# Spatial domain
x = np.linspace(0, 20, 500)

# Kink center
x0 = 10.0


# Stationary sine-Gordon kink
theta = 4 * np.arctan(
    np.exp(
        (omega0 / c0) * (x - x0)
    )
)

theta_antikink = 4 * np.arctan(
    np.exp(
        -(omega0 / c0) * (x - x0)
    )
)

# Plot
plt.figure(figsize=(7, 5))

plt.plot(x, theta, linewidth=2, label="Kink")
plt.plot(x, theta_antikink, linewidth=2, linestyle="--", label="Antikink")

plt.xlabel("Position x")
plt.ylabel(r"$\theta$")

plt.yticks(
    [0, np.pi, 2 * np.pi],
    [r"$0$", r"$\pi$", r"$2\pi$"]
)

plt.title("Stationary Sine-Gordon Kink and Antikink")

plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/kink_antikink_profiles.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()