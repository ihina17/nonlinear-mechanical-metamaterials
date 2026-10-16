import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.model.parameters import a, omega0, c0


# ============================================================
# Output folders
# ============================================================

os.makedirs("figures", exist_ok=True)
os.makedirs("animations", exist_ok=True)


# ============================================================
# Numerical parameters
# ============================================================

N = 400
dt = 0.001
t_final = 6.0

x = np.arange(N) * a
t = np.arange(0.0, t_final + dt, dt)

M = len(t)


# ============================================================
# Collision parameters
# ============================================================

v = 0.8 * c0

gamma = 1.0 / np.sqrt(
    1.0 - (v / c0) ** 2
)

beta = gamma * omega0 / c0

x1 = 15.0
x2 = 25.0


# ============================================================
# Discrete sine-Gordon solver
# ============================================================

def solve_sine_gordon(theta0, theta_t0, left_bc, right_bc):
    """
    Solve the discrete sine-Gordon equation using
    symplectic Euler time integration.
    """

    theta = np.zeros((N, M))
    theta_t = np.zeros((N, M))

    # Initial conditions
    theta[:, 0] = theta0
    theta_t[:, 0] = theta_t0

    # Boundary conditions
    theta[0, 0] = left_bc
    theta[-1, 0] = right_bc

    theta_t[0, 0] = 0.0
    theta_t[-1, 0] = 0.0

    # Time integration
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

        # Symplectic Euler:
        # update velocity first
        theta_t[:, n + 1] = (
            theta_t[:, n]
            + dt * force
        )

        # then update field
        theta[:, n + 1] = (
            theta[:, n]
            + dt * theta_t[:, n + 1]
        )

        # Enforce boundary conditions
        theta[0, n + 1] = left_bc
        theta[-1, n + 1] = right_bc

        theta_t[0, n + 1] = 0.0
        theta_t[-1, n + 1] = 0.0

    return theta, theta_t


# ============================================================
# Energy density
# ============================================================

def compute_energy_density(theta, theta_t):
    """
    Compute

        E = 1/2 theta_t^2
          + 1/2 c0^2 theta_x^2
          + omega0^2 (1 - cos(theta)).
    """

    theta_x = np.gradient(
        theta,
        a,
        axis=0
    )

    energy = (
        0.5 * theta_t**2
        + 0.5 * c0**2 * theta_x**2
        + omega0**2 * (1.0 - np.cos(theta))
    )

    return energy


# ============================================================
# Plot initial configuration
# ============================================================

def plot_initial_configuration(
    theta0,
    title,
    filename,
    yticks,
    yticklabels
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        x,
        theta0,
        linewidth=2
    )

    plt.xlabel("Position x")
    plt.ylabel(r"$\theta$")
    plt.title(title)

    plt.yticks(
        yticks,
        yticklabels
    )

    plt.grid()
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# Plot energy-density space-time map
# ============================================================

def plot_energy_density(
    energy,
    title,
    filename
):
    plt.figure(figsize=(9, 6))

    plt.contourf(
        x,
        t,
        energy.T,
        levels=80
    )

    plt.colorbar(
        label="Energy density"
    )

    plt.xlabel("Position x")
    plt.ylabel("Time t")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# Plot total energy
# ============================================================

def plot_total_energy(
    total_energy,
    title,
    filename
):
    plt.figure(figsize=(7, 5))

    plt.plot(
        t,
        total_energy,
        linewidth=2
    )

    plt.xlabel("Time t")
    plt.ylabel("Total energy")
    plt.title(title)

    plt.grid()
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# Animate energy density
# ============================================================

def animate_energy_density(
    energy,
    title_prefix,
    filename
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    line, = ax.plot(
        x,
        energy[:, 0],
        linewidth=2
    )

    ax.set_xlim(
        x[0],
        x[-1]
    )

    ax.set_ylim(
        0.0,
        1.1 * np.max(energy)
    )

    ax.set_xlabel("Position x")
    ax.set_ylabel("Energy density")
    ax.grid()

    title = ax.set_title(
        f"{title_prefix}, t = {t[0]:.2f}"
    )

    frame_indices = np.arange(
        0,
        M,
        20
    )

    def update(frame):

        n = frame_indices[frame]

        line.set_ydata(
            energy[:, n]
        )

        title.set_text(
            f"{title_prefix}, t = {t[n]:.2f}"
        )

        return line, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=30,
        blit=False
    )

    animation.save(
        filename,
        writer=PillowWriter(fps=25),
        dpi=120
    )

    plt.close(fig)


# ============================================================
# 1. KINK-KINK COLLISION
# ============================================================

# Kink 1: moving right
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


# Kink 2: moving left
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


# Combined initial condition
theta0_kk = kink1 + kink2

theta_t0_kk = (
    velocity1
    + velocity2
)


# Solve
theta_kk, theta_t_kk = solve_sine_gordon(
    theta0_kk,
    theta_t0_kk,
    left_bc=0.0,
    right_bc=4.0 * np.pi
)


# ------------------------------------------------------------
# Kink-kink energy
# ------------------------------------------------------------

energy_kk = compute_energy_density(
    theta_kk,
    theta_t_kk
)

total_energy_kk = (
    a * np.sum(
        energy_kk,
        axis=0
    )
)

energy_variation_kk = (
    (
        np.max(total_energy_kk)
        - np.min(total_energy_kk)
    )
    / np.mean(total_energy_kk)
    * 100
)

print(
    f"Kink-kink energy variation: "
    f"{energy_variation_kk:.4f}%"
)


# ------------------------------------------------------------
# Kink-kink figures
# ------------------------------------------------------------

plot_initial_configuration(
    theta0_kk,
    title="Initial Kink–Kink Configuration",
    filename="figures/initial_kink_kink.png",
    yticks=[
        0.0,
        2.0 * np.pi,
        4.0 * np.pi
    ],
    yticklabels=[
        r"$0$",
        r"$2\pi$",
        r"$4\pi$"
    ]
)


plot_energy_density(
    energy_kk,
    title="Kink–Kink Collision: Energy Density",
    filename="figures/energy_density_kink_kink.png"
)


plot_total_energy(
    total_energy_kk,
    title="Kink–Kink Collision: Total Energy",
    filename="figures/total_energy_kink_kink.png"
)


animate_energy_density(
    energy_kk,
    title_prefix="Kink–Kink Energy Density",
    filename="animations/energy_density_kink_kink.gif"
)


# ============================================================
# 2. KINK-ANTIKINK COLLISION
# ============================================================

# Kink: moving right
kink = 4.0 * np.arctan(
    np.exp(
        beta * (x - x1)
    )
)

kink_velocity = (
    -2.0
    * v
    * beta
    / np.cosh(beta * (x - x1))
)


# Antikink: moving left
antikink = 4.0 * np.arctan(
    np.exp(
        -beta * (x - x2)
    )
)

antikink_velocity = (
    -2.0
    * v
    * beta
    / np.cosh(beta * (x - x2))
)


# Combined initial condition
theta0_ka = (
    kink
    + antikink
    - 2.0 * np.pi
)

theta_t0_ka = (
    kink_velocity
    + antikink_velocity
)


# Solve
theta_ka, theta_t_ka = solve_sine_gordon(
    theta0_ka,
    theta_t0_ka,
    left_bc=0.0,
    right_bc=0.0
)


# ------------------------------------------------------------
# Kink-antikink energy
# ------------------------------------------------------------

energy_ka = compute_energy_density(
    theta_ka,
    theta_t_ka
)

total_energy_ka = (
    a * np.sum(
        energy_ka,
        axis=0
    )
)

energy_variation_ka = (
    (
        np.max(total_energy_ka)
        - np.min(total_energy_ka)
    )
    / np.mean(total_energy_ka)
    * 100
)

print(
    f"Kink-antikink energy variation: "
    f"{energy_variation_ka:.4f}%"
)


# ------------------------------------------------------------
# Kink-antikink figures
# ------------------------------------------------------------

plot_initial_configuration(
    theta0_ka,
    title="Initial Kink–Antikink Configuration",
    filename="figures/initial_kink_antikink.png",
    yticks=[
        0.0,
        np.pi,
        2.0 * np.pi
    ],
    yticklabels=[
        r"$0$",
        r"$\pi$",
        r"$2\pi$"
    ]
)


plot_energy_density(
    energy_ka,
    title="Kink–Antikink Collision: Energy Density",
    filename="figures/energy_density_kink_antikink.png"
)


plot_total_energy(
    total_energy_ka,
    title="Kink–Antikink Collision: Total Energy",
    filename="figures/total_energy_kink_antikink.png"
)


animate_energy_density(
    energy_ka,
    title_prefix="Kink–Antikink Energy Density",
    filename="animations/energy_density_kink_antikink.gif"
)


