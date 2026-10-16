import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.model.parameters import a, omega0, c0


# ============================================================
# Spatial domain
# ============================================================

N = 600

x = np.arange(N) * a


# ============================================================
# Breather parameters
# ============================================================

Omega = 0.8 * omega0

alpha = np.sqrt(
    omega0**2 - Omega**2
)

V = 0.5 * c0

gamma = 1.0 / np.sqrt(
    1.0 - (V / c0) ** 2
)


# Initial breather centers
x1 = 15.0
x2 = 25.0


# Start both near maximum amplitude
phi = np.pi / 2


# ============================================================
# Breather 1: moving right
# ============================================================

xi1 = gamma * (x - x1)

breather1 = 4.0 * np.arctan(
    (
        alpha * np.sin(phi)
    )
    /
    (
        Omega
        * np.cosh(
            (alpha / c0) * xi1
        )
    )
)


# ============================================================
# Breather 2: moving left
# ============================================================

xi2 = gamma * (x - x2)

breather2 = 4.0 * np.arctan(
    (
        alpha * np.sin(phi)
    )
    /
    (
        Omega
        * np.cosh(
            (alpha / c0) * xi2
        )
    )
)


# ============================================================
# Combined initial field
# ============================================================

theta0 = breather1 + breather2

# ============================================================
# Initial velocities
# ============================================================

# Left breather moves to the right
theta_t1 = -V * np.gradient(
    breather1,
    a
)

# Right breather moves to the left
theta_t2 = V * np.gradient(
    breather2,
    a
)

# Combined initial velocity
theta_t0 = theta_t1 + theta_t2

# ============================================================
# Time integration
# ============================================================

dt = 0.001
t_final = 20.0

t = np.arange(0.0, t_final + dt, dt)
M = len(t)


# Allocate arrays
theta = np.zeros((N, M))
theta_t = np.zeros((N, M))


# Initial conditions
theta[:, 0] = theta0
theta_t[:, 0] = theta_t0


# Fixed boundaries
theta[0, 0] = 0.0
theta[-1, 0] = 0.0

theta_t[0, 0] = 0.0
theta_t[-1, 0] = 0.0


# Symplectic Euler integration
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

    # Boundary conditions
    theta[0, n + 1] = 0.0
    theta[-1, n + 1] = 0.0

    theta_t[0, n + 1] = 0.0
    theta_t[-1, n + 1] = 0.0


# ============================================================
# Plot initial configuration
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    theta0,
    linewidth=2
)

plt.xlabel("Position x")
plt.ylabel(r"$\theta$")

plt.title(
    "Initial Breather–Breather Configuration"
)

plt.grid()
plt.tight_layout()

plt.savefig(
    "figures/breather_breather_initial.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    theta_t0,
    linewidth=2
)

plt.xlabel("Position x")
plt.ylabel(r"$\dot{\theta}$")
plt.title("Initial Breather Velocities")

plt.grid()
plt.tight_layout()

plt.show()

# ============================================================
# Space-time plot
# ============================================================

plt.figure(figsize=(9, 6))

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

plt.title(
    "Breather–Breather Collision"
)

plt.tight_layout()

plt.savefig(
    "figures/breather_breather_collision.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Breather-breather collision animation
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

line, = ax.plot(
    x,
    theta[:, 0],
    linewidth=2
)

ax.set_xlim(
    x[0],
    x[-1]
)

theta_max = np.max(np.abs(theta))

ax.set_ylim(
    -1.1 * theta_max,
    1.1 * theta_max
)

ax.set_xlabel("Position x")
ax.set_ylabel(r"$\theta$")
ax.grid()

title = ax.set_title(
    "Breather–Breather Collision"
)


# Use every 50th simulation step
frame_indices = np.arange(
    0,
    M,
    50
)


def update(frame):

    n = frame_indices[frame]

    line.set_ydata(
        theta[:, n]
    )

    title.set_text(
        f"Breather–Breather Collision, t = {t[n]:.2f}"
    )

    return line, title


animation = FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    interval=40,
    blit=False
)


animation.save(
    "animations/breather_breather_collision.gif",
    writer=PillowWriter(fps=25),
    dpi=120
)

plt.close(fig)

# ============================================================
# Energy density
# ============================================================

theta_x = np.gradient(
    theta,
    a,
    axis=0
)

energy_density = (
    0.5 * theta_t**2
    + 0.5 * c0**2 * theta_x**2
    + omega0**2 * (1.0 - np.cos(theta))
)

# ============================================================
# Energy-density space-time plot
# ============================================================

plt.figure(figsize=(9, 6))

plt.contourf(
    x,
    t,
    energy_density.T,
    levels=80
)

plt.colorbar(
    label="Energy density"
)

plt.xlabel("Position x")
plt.ylabel("Time t")

plt.title(
    "Breather–Breather Collision: Energy Density"
)

plt.tight_layout()

plt.savefig(
    "figures/breather_breather_energy_density.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()