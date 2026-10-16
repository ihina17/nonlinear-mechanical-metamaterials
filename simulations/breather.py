import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.model.parameters import a, omega0, c0


# ------------------------------------------------------------
# Spatial domain
# ------------------------------------------------------------

x = np.linspace(0, 40, 800)

# Breather center
x0 = 20.0


# ------------------------------------------------------------
# Breather parameters
# ------------------------------------------------------------

Omega = 0.2 * omega0

alpha = np.sqrt(
    omega0**2 - Omega**2
)


# ------------------------------------------------------------
# Choose one time
# ------------------------------------------------------------

t = 0.5


# ------------------------------------------------------------
# Analytical breather solution
# ------------------------------------------------------------

theta = 4 * np.arctan(
    (
        alpha * np.sin(Omega * t)
    )
    /
    (
        Omega
        * np.cosh(
            (alpha / c0) * (x - x0)
        )
    )
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    theta,
    linewidth=2
)

plt.xlabel("Position x")
plt.ylabel(r"$\theta$")

plt.title(
    "Stationary Sine-Gordon Breather"
)

plt.grid()
plt.tight_layout()

plt.savefig(
    "figures/breather_profile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# Breather animation
# ------------------------------------------------------------

os.makedirs("animations", exist_ok=True)

# One full oscillation period
T = 2 * np.pi / Omega

t_animation = np.linspace(
    0,
    T,
    200
)

fig, ax = plt.subplots(figsize=(8, 5))

line, = ax.plot(
    x,
    np.zeros_like(x),
    linewidth=2
)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(-2 * np.pi, 2 * np.pi)

ax.set_xlabel("Position x")
ax.set_ylabel(r"$\theta$")

ax.set_yticks(
    [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]
)

ax.set_yticklabels(
    [
        r"$-2\pi$",
        r"$-\pi$",
        r"$0$",
        r"$\pi$",
        r"$2\pi$"
    ]
)

ax.grid()

title = ax.set_title(
    "Stationary Sine-Gordon Breather"
)


def update(frame):

    current_time = t_animation[frame] + T / 4

    theta_breather = 4 * np.arctan(
        (
            alpha * np.sin(Omega * current_time)
        )
        /
        (
            Omega
            * np.cosh(
                (alpha / c0) * (x - x0)
            )
        )
    )

    line.set_ydata(theta_breather)

    title.set_text(
        f"Stationary Sine-Gordon Breather, "
        f"t = {current_time:.2f}"
    )

    return line, title


# animation = FuncAnimation(
#     fig,
#     update,
#     frames=len(t_animation),
#     interval=40,
#     blit=False
# )

# animation.save(
#     "animations/stationary_breather.gif",
#     writer=PillowWriter(fps=25),
#     dpi=120
# )

# plt.close(fig)

# ============================================================
# Discrete breather simulation
# ============================================================

# Numerical parameters
N = 400
dt = 0.001
t_final = 20.0

x_nodes = np.arange(N) * a
t_sim = np.arange(0.0, t_final + dt, dt)

M = len(t_sim)


# ------------------------------------------------------------
# Initial breather condition
# ------------------------------------------------------------

# Start at maximum amplitude
phi = np.pi / 2

theta_num = np.zeros((N, M))
theta_t_num = np.zeros((N, M))


theta_num[:, 0] = 4 * np.arctan(
    (
        alpha * np.sin(phi)
    )
    /
    (
        Omega
        * np.cosh(
            (alpha / c0) * (x_nodes - x0)
        )
    )
)

# At maximum amplitude, initial velocity is zero
theta_t_num[:, 0] = 0.0


# Fixed boundaries
theta_num[0, 0] = 0.0
theta_num[-1, 0] = 0.0

theta_t_num[0, 0] = 0.0
theta_t_num[-1, 0] = 0.0


# ------------------------------------------------------------
# Symplectic Euler time integration
# ------------------------------------------------------------

for n in range(M - 1):

    theta_xx = np.zeros(N)

    theta_xx[1:-1] = (
        theta_num[2:, n]
        - 2.0 * theta_num[1:-1, n]
        + theta_num[:-2, n]
    ) / a**2

    force = (
        c0**2 * theta_xx
        - omega0**2 * np.sin(theta_num[:, n])
    )

    # Fixed boundaries
    force[0] = 0.0
    force[-1] = 0.0


    # Update velocity
    theta_t_num[:, n + 1] = (
        theta_t_num[:, n]
        + dt * force
    )


    # Update field
    theta_num[:, n + 1] = (
        theta_num[:, n]
        + dt * theta_t_num[:, n + 1]
    )


    # Enforce boundary conditions
    theta_num[0, n + 1] = 0.0
    theta_num[-1, n + 1] = 0.0

    theta_t_num[0, n + 1] = 0.0
    theta_t_num[-1, n + 1] = 0.0


    # ============================================================
# Numerical vs analytical center oscillation
# ============================================================

# Find the lattice node closest to the breather center
center_index = np.argmin(
    np.abs(x_nodes - x0)
)


# Numerical solution at the center
theta_center_num = theta_num[
    center_index,
    :
]


# Analytical solution at the center
theta_center_exact = 4 * np.arctan(
    (
        alpha
        * np.sin(Omega * t_sim + phi)
    )
    / Omega
)


# ------------------------------------------------------------
# Plot comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    t_sim,
    theta_center_num,
    linewidth=2,
    label="Discrete numerical"
)

plt.plot(
    t_sim,
    theta_center_exact,
    "--",
    linewidth=2,
    label="Analytical"
)

plt.xlabel("Time t")
plt.ylabel(r"$\theta(x_0,t)$")

plt.title(
    "Breather Center Oscillation: Numerical vs Analytical"
)

plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/breather_center_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()