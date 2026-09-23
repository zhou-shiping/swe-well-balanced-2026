"""Vertically resolved reference solver for the smooth two-dimensional benchmark.

The code advances one column height and layer-centered horizontal momenta on a
uniform (x, y, zeta) finite-volume grid.  A continuity-based recurrence builds
the mapped vertical mass flux at every Runge--Kutta stage, so every vertical
layer receives exactly the same semidiscrete height update.
"""

from __future__ import annotations

import math

import numpy as np


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    positive = (a > 0.0) & (b > 0.0) & (c > 0.0)
    negative = (a < 0.0) & (b < 0.0) & (c < 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(positive, magnitude, np.where(negative, -magnitude, 0.0))


def _mc_difference(values: np.ndarray, axis: int) -> np.ndarray:
    backward = values - np.roll(values, 1, axis=axis)
    forward = np.roll(values, -1, axis=axis) - values
    centered = 0.5 * (backward + forward)
    return _minmod3(1.5 * backward, centered, 1.5 * forward)


def shifted_legendre(zeta: np.ndarray, degree: int) -> np.ndarray:
    """Return the sign convention used in the moment expansion."""
    if degree == 0:
        return np.ones_like(zeta)
    if degree == 1:
        return 1.0 - 2.0 * zeta
    if degree == 2:
        return 1.0 - 6.0 * zeta + 6.0 * zeta**2
    raise ValueError("this benchmark projects only degrees 0, 1, and 2")


def initial_state(
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cell centers, height, and layer momenta for the common profile."""
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    zeta = (np.arange(nz) + 0.5) / nz
    xx, yy = np.meshgrid(x, y, indexing="ij")

    height = 1.0 + 0.05 * np.cos(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    mean_u = 0.2 + 0.02 * np.sin(2.0 * np.pi * yy)
    mean_v = 0.1 + 0.02 * np.sin(2.0 * np.pi * xx)
    alpha_1 = 0.03 * (1.0 + 0.2 * np.cos(2.0 * np.pi * xx))
    beta_1 = 0.02 * (1.0 + 0.2 * np.sin(2.0 * np.pi * yy))
    alpha_2 = 0.01 * np.sin(2.0 * np.pi * (xx + yy))
    beta_2 = 0.015 * np.cos(2.0 * np.pi * (xx - yy))

    phi_1 = shifted_legendre(zeta, 1)
    phi_2 = shifted_legendre(zeta, 2)
    velocity_u = (
        mean_u[..., None]
        + alpha_1[..., None] * phi_1
        + alpha_2[..., None] * phi_2
    )
    velocity_v = (
        mean_v[..., None]
        + beta_1[..., None] * phi_1
        + beta_2[..., None] * phi_2
    )
    momentum_u = height[..., None] * velocity_u
    momentum_v = height[..., None] * velocity_v
    return x, y, zeta, height, np.stack((momentum_u, momentum_v), axis=-1)


def _horizontal_fluxes(
    height: np.ndarray,
    momentum: np.ndarray,
    gravity: float,
    axis: int,
) -> np.ndarray:
    """Return a common-speed Rusanov flux for every vertical layer."""
    height_difference = _mc_difference(height, axis)
    momentum_difference = _mc_difference(momentum, axis)
    height_left = height + 0.5 * height_difference
    height_right = np.roll(height, -1, axis=axis) - 0.5 * np.roll(
        height_difference, -1, axis=axis
    )
    momentum_left = momentum + 0.5 * momentum_difference
    momentum_right = np.roll(momentum, -1, axis=axis) - 0.5 * np.roll(
        momentum_difference, -1, axis=axis
    )
    if np.any(height_left <= 0.0) or np.any(height_right <= 0.0):
        raise FloatingPointError("MUSCL reconstruction produced a non-positive depth")

    velocity_left = momentum_left / height_left[..., None, None]
    velocity_right = momentum_right / height_right[..., None, None]
    normal_component = axis
    normal_left = velocity_left[..., normal_component]
    normal_right = velocity_right[..., normal_component]
    wave_speed = np.maximum(
        np.max(np.abs(normal_left), axis=2) + np.sqrt(gravity * height_left),
        np.max(np.abs(normal_right), axis=2) + np.sqrt(gravity * height_right),
    )

    number_of_layers = momentum.shape[2]
    conservative_left = np.empty(momentum.shape[:-1] + (3,))
    conservative_right = np.empty_like(conservative_left)
    conservative_left[..., 0] = np.broadcast_to(
        height_left[..., None], height_left.shape + (number_of_layers,)
    )
    conservative_right[..., 0] = np.broadcast_to(
        height_right[..., None], height_right.shape + (number_of_layers,)
    )
    conservative_left[..., 1:] = momentum_left
    conservative_right[..., 1:] = momentum_right

    flux_left = np.empty_like(conservative_left)
    flux_right = np.empty_like(conservative_right)
    flux_left[..., 0] = momentum_left[..., normal_component]
    flux_right[..., 0] = momentum_right[..., normal_component]
    for component in range(2):
        flux_left[..., 1 + component] = (
            momentum_left[..., component] * normal_left
        )
        flux_right[..., 1 + component] = (
            momentum_right[..., component] * normal_right
        )
    flux_left[..., 1 + normal_component] += 0.5 * gravity * height_left[..., None] ** 2
    flux_right[..., 1 + normal_component] += 0.5 * gravity * height_right[..., None] ** 2

    return 0.5 * (flux_left + flux_right) - 0.5 * wave_speed[..., None, None] * (
        conservative_right - conservative_left
    )


def _vertical_mass_flux(
    mass_divergence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build W=h*omega and return it with the common column divergence."""
    number_of_layers = mass_divergence.shape[2]
    mean_divergence = np.mean(mass_divergence, axis=2)
    vertical_mass_flux = np.zeros(mass_divergence.shape[:2] + (number_of_layers + 1,))
    for layer in range(number_of_layers):
        vertical_mass_flux[..., layer + 1] = (
            vertical_mass_flux[..., layer]
            + (mean_divergence - mass_divergence[..., layer]) / number_of_layers
        )
    return vertical_mass_flux, mean_divergence


def _vertical_velocity_flux(
    height: np.ndarray,
    momentum: np.ndarray,
    vertical_mass_flux: np.ndarray,
) -> np.ndarray:
    """Upwind the layer velocity with the continuity-compatible mass flux."""
    velocity = momentum / height[..., None, None]
    slope = np.zeros_like(velocity)
    if velocity.shape[2] > 2:
        backward = velocity[..., 1:-1, :] - velocity[..., :-2, :]
        forward = velocity[..., 2:, :] - velocity[..., 1:-1, :]
        slope[..., 1:-1, :] = _minmod3(
            1.5 * backward,
            0.5 * (backward + forward),
            1.5 * forward,
        )

    vertical_flux = np.zeros(momentum.shape[:2] + (momentum.shape[2] + 1, 2))
    if momentum.shape[2] > 1:
        lower_trace = velocity[..., :-1, :] + 0.5 * slope[..., :-1, :]
        upper_trace = velocity[..., 1:, :] - 0.5 * slope[..., 1:, :]
        interior_mass_flux = vertical_mass_flux[..., 1:-1]
        upwind_velocity = np.where(
            interior_mass_flux[..., None] >= 0.0,
            lower_trace,
            upper_trace,
        )
        vertical_flux[..., 1:-1, :] = interior_mass_flux[..., None] * upwind_velocity
    return vertical_flux


def spatial_operator(
    height: np.ndarray,
    momentum: np.ndarray,
    gravity: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux_x = _horizontal_fluxes(height, momentum, gravity, axis=0)
    flux_y = _horizontal_fluxes(height, momentum, gravity, axis=1)
    divergence_x = (flux_x - np.roll(flux_x, 1, axis=0)) / dx
    divergence_y = (flux_y - np.roll(flux_y, 1, axis=1)) / dy
    horizontal_divergence = divergence_x + divergence_y

    vertical_mass_flux, mean_mass_divergence = _vertical_mass_flux(
        horizontal_divergence[..., 0]
    )
    vertical_momentum_flux = _vertical_velocity_flux(
        height, momentum, vertical_mass_flux
    )
    dzeta = 1.0 / momentum.shape[2]
    vertical_divergence = (
        vertical_momentum_flux[..., 1:, :] - vertical_momentum_flux[..., :-1, :]
    ) / dzeta
    height_rate = -mean_mass_divergence
    momentum_rate = -horizontal_divergence[..., 1:] - vertical_divergence
    return height_rate, momentum_rate, vertical_mass_flux


def stable_timestep(
    height: np.ndarray,
    momentum: np.ndarray,
    vertical_mass_flux: np.ndarray,
    gravity: float,
    dx: float,
    dy: float,
    cfl: float,
) -> float:
    velocity = momentum / height[..., None, None]
    speed_x = float(np.max(np.abs(velocity[..., 0]) + np.sqrt(gravity * height)[..., None]))
    speed_y = float(np.max(np.abs(velocity[..., 1]) + np.sqrt(gravity * height)[..., None]))
    omega = vertical_mass_flux / height[..., None]
    speed_zeta = float(np.max(np.abs(omega)))
    dzeta = 1.0 / momentum.shape[2]
    return cfl / (speed_x / dx + speed_y / dy + speed_zeta / dzeta)


def _validated(height: np.ndarray, momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(height)) or not np.all(np.isfinite(momentum)):
        raise FloatingPointError("the vertically resolved state contains a non-finite value")
    if float(np.min(height)) <= 0.0:
        raise FloatingPointError("the vertically resolved state lost positive depth")
    return height, momentum


def ssprk3_step(
    height: np.ndarray,
    momentum: np.ndarray,
    dt: float,
    gravity: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    height_rate_0, momentum_rate_0, _ = spatial_operator(
        height, momentum, gravity, dx, dy
    )
    height_1, momentum_1 = _validated(
        height + dt * height_rate_0,
        momentum + dt * momentum_rate_0,
    )
    height_rate_1, momentum_rate_1, _ = spatial_operator(
        height_1, momentum_1, gravity, dx, dy
    )
    height_2, momentum_2 = _validated(
        0.75 * height + 0.25 * (height_1 + dt * height_rate_1),
        0.75 * momentum + 0.25 * (momentum_1 + dt * momentum_rate_1),
    )
    height_rate_2, momentum_rate_2, _ = spatial_operator(
        height_2, momentum_2, gravity, dx, dy
    )
    return _validated(
        height / 3.0 + 2.0 * (height_2 + dt * height_rate_2) / 3.0,
        momentum / 3.0 + 2.0 * (momentum_2 + dt * momentum_rate_2) / 3.0,
    )


def project_moments(
    height: np.ndarray,
    momentum: np.ndarray,
    maximum_order: int = 2,
) -> np.ndarray:
    """Project the resolved velocity with the solver's midpoint quadrature."""
    number_of_layers = momentum.shape[2]
    zeta = (np.arange(number_of_layers) + 0.5) / number_of_layers
    velocity = momentum / height[..., None, None]
    projected = np.empty(height.shape + (3 + 2 * maximum_order,))
    projected[..., 0] = height
    projected[..., 1] = height * np.mean(velocity[..., 0], axis=2)
    projected[..., 2] = height * np.mean(velocity[..., 1], axis=2)
    for degree in range(1, maximum_order + 1):
        phi = shifted_legendre(zeta, degree)
        coefficient_u = (2 * degree + 1) * np.mean(velocity[..., 0] * phi, axis=2)
        coefficient_v = (2 * degree + 1) * np.mean(velocity[..., 1] * phi, axis=2)
        projected[..., 1 + 2 * degree] = height * coefficient_u
        projected[..., 2 + 2 * degree] = height * coefficient_v
    return projected


def solve(
    nx: int,
    ny: int,
    nz: int,
    final_time: float = 0.1,
    gravity: float = 1.0,
    cfl: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    x, y, zeta, height, momentum = initial_state(nx, ny, nz)
    initial_mass = float(np.mean(height))
    dx = 1.0 / nx
    dy = 1.0 / ny
    time = 0.0
    steps = 0
    maximum_top_flux = 0.0

    while time < final_time:
        _, _, vertical_mass_flux = spatial_operator(height, momentum, gravity, dx, dy)
        maximum_top_flux = max(
            maximum_top_flux,
            float(np.max(np.abs(vertical_mass_flux[..., -1]))),
        )
        dt = min(
            stable_timestep(
                height, momentum, vertical_mass_flux, gravity, dx, dy, cfl
            ),
            final_time - time,
        )
        height, momentum = ssprk3_step(height, momentum, dt, gravity, dx, dy)
        time += dt
        steps += 1

    final_mass = float(np.mean(height))
    diagnostics: dict[str, float | int] = {
        "steps": steps,
        "final_time": time,
        "minimum_depth": float(np.min(height)),
        "maximum_depth": float(np.max(height)),
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "relative_mass_defect": abs(final_mass - initial_mass) / abs(initial_mass),
        "maximum_top_vertical_mass_flux": maximum_top_flux,
        "maximum_velocity": float(np.max(np.abs(momentum / height[..., None, None]))),
    }
    if not math.isclose(time, final_time, rel_tol=0.0, abs_tol=2.0e-14):
        raise RuntimeError("the time integrator did not land on the requested final time")
    return x, y, zeta, height, momentum, diagnostics


def swap_xy_state(
    height: np.ndarray,
    momentum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    swapped_height = np.swapaxes(height, 0, 1).copy()
    swapped_momentum = np.swapaxes(momentum, 0, 1).copy()
    swapped_momentum[..., [0, 1]] = swapped_momentum[..., [1, 0]]
    return swapped_height, swapped_momentum
