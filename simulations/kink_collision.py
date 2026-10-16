import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.model.parameters import a, omega0, c0


# ============================================================
# Numerical parameters
# ============================================================

N = 400
dt = 0.001
t_final = 6.0

x = np.arange(N) * a
t = np.arange(0, t_final + dt, dt)

M = len(t)


# ============================================================
# Soliton parameters
# ============================================================

v = 0.8 * c0

gamma = 1.0 / np.sqrt(
    1.0 - (v / c0) ** 2
)

beta = gamma * omega0 / c0


# ============================================================
# 1. KINK-KINK COLLISION
# ============================================================

# Initial positions
x1 = 15.0
x2 = 25.0


# ------------------------------------------------------------
# Initial kink 1: moving right
# ------------------------------------------------------------

kink1 = 4.0 * np.arctan(
    np.exp(
        beta * (x - x1)
    )
)

velocity1 = (
    -2.0
    * v
    * beta
    / np.cosh(beta * (x - x1))
)


# ------------------------------------------------------------
# Initial kink 2: moving left
# ------------------------------------------------------------

kink2 = 4.0 * np.arctan(
    np.exp(
        beta * (x - x2)
    )
)

velocity2 = (
    2.0
    * v
    * beta
    / np.cosh(beta * (x - x2))
)


# ------------------------------------------------------------
# Allocate arrays
# ------------------------------------------------------------

theta = np.zeros((N, M))
theta_t = np.zeros((N, M))


# Initial condition
theta[:, 0] = kink1 + kink2

theta_t[:, 0] = (
    velocity1
    + velocity2
)


# Fixed boundary conditions
theta[0, 0] = 0.0
theta[-1, 0] = 4.0 * np.pi

theta_t[0, 0] = 0.0
theta_t[-1, 0] = 0.0


# ------------------------------------------------------------
# Time integration: symplectic Euler
# ------------------------------------------------------------

for n in range(M - 1):

    theta_xx = np.zeros(N)

    theta_xx[1:-1] = (
        theta[2:, n]
        - 2.0 * theta[1:-1, n]
        + theta[:-2, n]
    ) / a**2

    force = (
        c0**2 * theta_xx
        - omega0**2 * np.sin(theta[:, n])
    )

    # Fixed boundaries
    force[0] = 0.0
    force[-1] = 0.0

    # Update velocity
    theta_t[:, n + 1] = (
        theta_t[:, n]
        + dt * force
    )

    # Update field
    theta[:, n + 1] = (
        theta[:, n]
        + dt * theta_t[:, n + 1]
    )

    # Enforce boundary conditions
    theta[0, n + 1] = 0.0
    theta[-1, n + 1] = 4.0 * np.pi

    theta_t[0, n + 1] = 0.0
    theta_t[-1, n + 1] = 0.0


# ------------------------------------------------------------
# Initial kink-kink configuration
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    theta[:, 0],
    linewidth=2
)

plt.xlabel("Position x")
plt.ylabel(r"$\theta$")

plt.yticks(
    [0, 2 * np.pi, 4 * np.pi],
    [r"$0$", r"$2\pi$", r"$4\pi$"]
)

plt.title("Initial Kink–Kink Configuration")

plt.grid()
plt.tight_layout()

plt.savefig(
    "figures/kink_kink_initial.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Raw field space-time plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.contourf(
    x,
    t,
    theta.T,
    levels=80
)

plt.colorbar(
    label=r"$\theta$"
)

plt.xlabel("Position x")
plt.ylabel("Time t")

plt.title("Kink–Kink Collision")

plt.tight_layout()

plt.savefig(
    "figures/kink_kink_field.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Localized kink cores
# ------------------------------------------------------------

theta_x = np.gradient(
    theta,
    a,
    axis=0
)

kink_core = np.abs(theta_x)


plt.figure(figsize=(8, 6))

plt.contourf(
    x,
    t,
    kink_core.T,
    levels=80
)

plt.colorbar(
    label=r"$|\partial \theta / \partial x|$"
)

plt.xlabel("Position x")
plt.ylabel("Time t")

plt.title(
    "Kink–Kink Collision: Localized Soliton Cores"
)

plt.tight_layout()

plt.savefig(
    "figures/kink_kink_collision.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Kink-kink collision animation
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

line, = ax.plot(
    x,
    kink_core[:, 0],
    linewidth=2
)

ax.set_xlim(
    x[0],
    x[-1]
)

ax.set_ylim(
    0,
    1.1 * np.max(kink_core)
)

ax.set_xlabel("Position x")

ax.set_ylabel(
    r"$|\partial \theta / \partial x|$"
)

ax.grid()

title = ax.set_title(
    "Kink–Kink Collision"
)


frame_indices = np.arange(
    0,
    M,
    20
)


def update_kink_kink(frame):

    n = frame_indices[frame]

    line.set_ydata(
        kink_core[:, n]
    )

    title.set_text(
        f"Kink–Kink Collision, t = {t[n]:.2f}"
    )

    return line, title


animation = FuncAnimation(
    fig,
    update_kink_kink,
    frames=len(frame_indices),
    interval=30,
    blit=False
)

animation.save(
    "animations/kink_kink_collision.gif",
    writer=PillowWriter(fps=25),
    dpi=120
)

plt.close(fig)


# ============================================================
# 2. KINK-ANTIKINK COLLISION
# ============================================================

# Initial positions
x_kink = 15.0
x_antikink = 25.0


# ------------------------------------------------------------
# Kink: moving right
# ------------------------------------------------------------

kink = 4.0 * np.arctan(
    np.exp(
        beta * (x - x_kink)
    )
)

kink_velocity = (
    -2.0
    * v
    * beta
    / np.cosh(beta * (x - x_kink))
)


# ------------------------------------------------------------
# Antikink: moving left
# ------------------------------------------------------------

antikink = 4.0 * np.arctan(
    np.exp(
        -beta * (x - x_antikink)
    )
)

antikink_velocity = (
    -2.0
    * v
    * beta
    / np.cosh(beta * (x - x_antikink))
)


# ------------------------------------------------------------
# Allocate arrays
# ------------------------------------------------------------

theta_ka = np.zeros((N, M))
theta_t_ka = np.zeros((N, M))


# Combined kink-antikink initial condition
theta_ka[:, 0] = (
    kink
    + antikink
    - 2.0 * np.pi
)

theta_t_ka[:, 0] = (
    kink_velocity
    + antikink_velocity
)


# Same vacuum at both boundaries
theta_ka[0, 0] = 0.0
theta_ka[-1, 0] = 0.0

theta_t_ka[0, 0] = 0.0
theta_t_ka[-1, 0] = 0.0


# ------------------------------------------------------------
# Initial kink-antikink configuration
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    theta_ka[:, 0],
    linewidth=2
)

plt.xlabel("Position x")
plt.ylabel(r"$\theta$")

plt.yticks(
    [0, np.pi, 2 * np.pi],
    [r"$0$", r"$\pi$", r"$2\pi$"]
)

plt.title(
    "Initial Kink–Antikink Configuration"
)

plt.grid()
plt.tight_layout()

plt.savefig(
    "figures/kink_antikink_initial.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Time integration: kink-antikink collision
# ------------------------------------------------------------

for n in range(M - 1):

    theta_xx_ka = np.zeros(N)

    theta_xx_ka[1:-1] = (
        theta_ka[2:, n]
        - 2.0 * theta_ka[1:-1, n]
        + theta_ka[:-2, n]
    ) / a**2

    force_ka = (
        c0**2 * theta_xx_ka
        - omega0**2 * np.sin(theta_ka[:, n])
    )

    # Fixed boundaries
    force_ka[0] = 0.0
    force_ka[-1] = 0.0

    # Update velocity
    theta_t_ka[:, n + 1] = (
        theta_t_ka[:, n]
        + dt * force_ka
    )

    # Update field
    theta_ka[:, n + 1] = (
        theta_ka[:, n]
        + dt * theta_t_ka[:, n + 1]
    )

    # Enforce boundary conditions
    theta_ka[0, n + 1] = 0.0
    theta_ka[-1, n + 1] = 0.0

    theta_t_ka[0, n + 1] = 0.0
    theta_t_ka[-1, n + 1] = 0.0


# ------------------------------------------------------------
# Localized kink and antikink cores
# ------------------------------------------------------------

theta_x_ka = np.gradient(
    theta_ka,
    a,
    axis=0
)

core_ka = np.abs(theta_x_ka)


plt.figure(figsize=(8, 6))

plt.contourf(
    x,
    t,
    core_ka.T,
    levels=80
)

plt.colorbar(
    label=r"$|\partial \theta / \partial x|$"
)

plt.xlabel("Position x")
plt.ylabel("Time t")

plt.title(
    "Kink–Antikink Collision: Localized Soliton Cores"
)

plt.tight_layout()

plt.savefig(
    "figures/kink_antikink_collision.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# Kink-antikink field animation
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

line, = ax.plot(
    x,
    theta_ka[:, 0],
    linewidth=2
)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.2, 2 * np.pi + 0.2)

ax.set_xlabel("Position x")
ax.set_ylabel(r"$\theta$")

ax.set_yticks(
    [0, np.pi, 2 * np.pi]
)

ax.set_yticklabels(
    [r"$0$", r"$\pi$", r"$2\pi$"]
)

ax.grid()

title = ax.set_title(
    "Kink–Antikink Collision"
)

frame_indices_ka = np.arange(
    0,
    M,
    20
)


def update_kink_antikink(frame):

    n = frame_indices_ka[frame]

    line.set_ydata(
        theta_ka[:, n]
    )

    title.set_text(
        f"Kink–Antikink Collision, t = {t[n]:.2f}"
    )

    return line, title


animation_ka = FuncAnimation(
    fig,
    update_kink_antikink,
    frames=len(frame_indices_ka),
    interval=30,
    blit=False
)

animation_ka.save(
    "animations/kink_antikink_collision.gif",
    writer=PillowWriter(fps=25),
    dpi=120
)

plt.close(fig)