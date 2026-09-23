"""Second-order finite-volume solver for the zero-moment radial collapse.

The state ordering is

    h, h*u, h*v, h*alpha_1, h*beta_1, ...

where ``x`` and ``z`` are the two horizontal coordinates and ``v`` is the
velocity in the horizontal z direction.  The implementation retains the
frictionless flat-bottom G-SWLME fluxes for moment orders 0, 1, and 2.  In the
Example 5 initial state every moment is zero, so each order stays on the SWE
subsystem and should give the same depth and mean velocity.
"""

from __future__ import annotations

import math

import numpy as np


DOMAIN_LENGTH_M = 100.0
GRAVITY_M_PER_S2 = 9.81
CENTER_M = 50.0
RADIUS_M = 15.0
INNER_DEPTH_M = 1.5
OUTER_DEPTH_M = 1.0
OUTPUT_TIMES_S = (0.0, 1.0, 2.0, 3.0)


def _check_order(order: int) -> None:
    if order not in (0, 1, 2):
        raise ValueError("this test supports moment orders 0, 1, and 2")


def initial_state(
    nx: int,
    nz: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the cell-centered radial collapse state in physical units."""

    _check_order(order)
    if nx < 4 or nz < 4:
        raise ValueError("nx and nz must both be at least four")
    x = (np.arange(nx, dtype=float) + 0.5) * DOMAIN_LENGTH_M / nx
    z = (np.arange(nz, dtype=float) + 0.5) * DOMAIN_LENGTH_M / nz
    xx, zz = np.meshgrid(x, z, indexing="ij")
    inside = (xx - CENTER_M) ** 2 + (zz - CENTER_M) ** 2 <= RADIUS_M**2

    state = np.zeros((nx, nz, 3 + 2 * order), dtype=np.float64)
    state[..., 0] = np.where(inside, INNER_DEPTH_M, OUTER_DEPTH_M)
    return x, z, state


def primitive(
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return height, mean velocities, and paired moment coefficients."""

    height = state[..., 0]
    if np.any(height <= 0.0):
        raise FloatingPointError("the state contains non-positive water depth")
    mean_u = state[..., 1] / height
    mean_v = state[..., 2] / height
    alpha = state[..., 3::2] / height[..., None]
    beta = state[..., 4::2] / height[..., None]
    return height, mean_u, mean_v, alpha, beta


def flux_x(state: np.ndarray, gravity: float = GRAVITY_M_PER_S2) -> np.ndarray:
    """Return the conservative G-SWLME flux in the horizontal x direction."""

    height, mean_u, mean_v, alpha, beta = primitive(state)
    flux = np.zeros_like(state)
    flux[..., 0] = state[..., 1]
    if alpha.shape[-1]:
        denominators = 2.0 * np.arange(1, alpha.shape[-1] + 1) + 1.0
        normal_energy = np.sum(alpha * alpha / denominators, axis=-1)
        cross_energy = np.sum(alpha * beta / denominators, axis=-1)
    else:
        normal_energy = 0.0
        cross_energy = 0.0
    flux[..., 1] = (
        height * (mean_u * mean_u + normal_energy)
        + 0.5 * gravity * height**2
    )
    flux[..., 2] = height * (mean_u * mean_v + cross_energy)
    for moment in range(alpha.shape[-1]):
        alpha_index = 3 + 2 * moment
        beta_index = alpha_index + 1
        flux[..., alpha_index] = 2.0 * height * mean_u * alpha[..., moment]
        flux[..., beta_index] = height * (
            mean_u * beta[..., moment] + mean_v * alpha[..., moment]
        )
    return flux


def flux_z(state: np.ndarray, gravity: float = GRAVITY_M_PER_S2) -> np.ndarray:
    """Return the conservative G-SWLME flux in the horizontal z direction."""

    height, mean_u, mean_v, alpha, beta = primitive(state)
    flux = np.zeros_like(state)
    flux[..., 0] = state[..., 2]
    if beta.shape[-1]:
        denominators = 2.0 * np.arange(1, beta.shape[-1] + 1) + 1.0
        normal_energy = np.sum(beta * beta / denominators, axis=-1)
        cross_energy = np.sum(alpha * beta / denominators, axis=-1)
    else:
        normal_energy = 0.0
        cross_energy = 0.0
    flux[..., 1] = height * (mean_u * mean_v + cross_energy)
    flux[..., 2] = (
        height * (mean_v * mean_v + normal_energy)
        + 0.5 * gravity * height**2
    )
    for moment in range(beta.shape[-1]):
        alpha_index = 3 + 2 * moment
        beta_index = alpha_index + 1
        flux[..., alpha_index] = height * (
            mean_u * beta[..., moment] + mean_v * alpha[..., moment]
        )
        flux[..., beta_index] = 2.0 * height * mean_v * beta[..., moment]
    return flux


def max_speed_x(
    state: np.ndarray, gravity: float = GRAVITY_M_PER_S2
) -> np.ndarray:
    height, mean_u, _, alpha, _ = primitive(state)
    if alpha.shape[-1]:
        denominators = 2.0 * np.arange(1, alpha.shape[-1] + 1) + 1.0
        moment_energy = np.sum(alpha * alpha / denominators, axis=-1)
    else:
        moment_energy = 0.0
    return np.abs(mean_u) + np.sqrt(gravity * height + 3.0 * moment_energy)


def max_speed_z(
    state: np.ndarray, gravity: float = GRAVITY_M_PER_S2
) -> np.ndarray:
    height, _, mean_v, _, beta = primitive(state)
    if beta.shape[-1]:
        denominators = 2.0 * np.arange(1, beta.shape[-1] + 1) + 1.0
        moment_energy = np.sum(beta * beta / denominators, axis=-1)
    else:
        moment_energy = 0.0
    return np.abs(mean_v) + np.sqrt(gravity * height + 3.0 * moment_energy)


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    positive = (a > 0.0) & (b > 0.0) & (c > 0.0)
    negative = (a < 0.0) & (b < 0.0) & (c < 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(positive, magnitude, np.where(negative, -magnitude, 0.0))


def reflected_ghost_state(state: np.ndarray) -> np.ndarray:
    """Add one impermeable free-slip ghost layer on all four sides."""

    nx, nz, variable_count = state.shape
    padded = np.empty((nx + 2, nz + 2, variable_count), dtype=state.dtype)
    padded[1:-1, 1:-1] = state

    padded[0, 1:-1] = state[0]
    padded[-1, 1:-1] = state[-1]
    padded[0, 1:-1, 1::2] *= -1.0
    padded[-1, 1:-1, 1::2] *= -1.0

    padded[1:-1, 0] = state[:, 0]
    padded[1:-1, -1] = state[:, -1]
    padded[1:-1, 0, 2::2] *= -1.0
    padded[1:-1, -1, 2::2] *= -1.0
    return padded


def _limited_slopes(padded: np.ndarray, axis: int) -> np.ndarray:
    """Return MC-limited cell differences with zero wall-adjacent slopes."""

    slopes = np.zeros_like(padded)
    if axis == 0:
        backward = padded[1:-1, 1:-1] - padded[:-2, 1:-1]
        forward = padded[2:, 1:-1] - padded[1:-1, 1:-1]
        centered = 0.5 * (backward + forward)
        slopes[1:-1, 1:-1] = _minmod3(
            1.5 * backward, centered, 1.5 * forward
        )
        slopes[1, 1:-1] = 0.0
        slopes[-2, 1:-1] = 0.0
    elif axis == 1:
        backward = padded[1:-1, 1:-1] - padded[1:-1, :-2]
        forward = padded[1:-1, 2:] - padded[1:-1, 1:-1]
        centered = 0.5 * (backward + forward)
        slopes[1:-1, 1:-1] = _minmod3(
            1.5 * backward, centered, 1.5 * forward
        )
        slopes[1:-1, 1] = 0.0
        slopes[1:-1, -2] = 0.0
    else:
        raise ValueError("the horizontal axis must be 0 or 1")
    return slopes


def _rusanov_fluxes(
    state: np.ndarray,
    gravity: float,
    axis: int,
) -> np.ndarray:
    padded = reflected_ghost_state(state)
    slopes = _limited_slopes(padded, axis)
    if axis == 0:
        left = padded[:-1, 1:-1] + 0.5 * slopes[:-1, 1:-1]
        right = padded[1:, 1:-1] - 0.5 * slopes[1:, 1:-1]
        physical_left = flux_x(left, gravity)
        physical_right = flux_x(right, gravity)
        speed = np.maximum(
            max_speed_x(left, gravity), max_speed_x(right, gravity)
        )
    elif axis == 1:
        left = padded[1:-1, :-1] + 0.5 * slopes[1:-1, :-1]
        right = padded[1:-1, 1:] - 0.5 * slopes[1:-1, 1:]
        physical_left = flux_z(left, gravity)
        physical_right = flux_z(right, gravity)
        speed = np.maximum(
            max_speed_z(left, gravity), max_speed_z(right, gravity)
        )
    else:
        raise ValueError("the horizontal axis must be 0 or 1")
    if np.any(left[..., 0] <= 0.0) or np.any(right[..., 0] <= 0.0):
        raise FloatingPointError("MUSCL reconstruction produced non-positive depth")
    return 0.5 * (physical_left + physical_right) - 0.5 * speed[..., None] * (
        right - left
    )


def nonconservative_product(
    state: np.ndarray,
    dx: float,
    dz: float,
) -> np.ndarray:
    """Return the G-SWLME nonconservative transport product."""

    _, mean_u, mean_v, alpha, beta = primitive(state)
    padded = reflected_ghost_state(state)
    derivative_x = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * dx)
    derivative_z = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * dz)
    product = np.zeros_like(state)
    for moment in range(alpha.shape[-1]):
        denominator = 2.0 * (moment + 1) + 1.0
        alpha_index = 3 + 2 * moment
        beta_index = alpha_index + 1
        product[..., 2] += (
            -beta[..., moment] * derivative_x[..., alpha_index]
            + alpha[..., moment] * derivative_x[..., beta_index]
        ) / denominator
        product[..., 1] += (
            beta[..., moment] * derivative_z[..., alpha_index]
            - alpha[..., moment] * derivative_z[..., beta_index]
        ) / denominator
        product[..., alpha_index] += (
            -mean_u * derivative_x[..., alpha_index]
            - mean_u * derivative_z[..., beta_index]
        )
        product[..., beta_index] += (
            -mean_v * derivative_x[..., alpha_index]
            - mean_v * derivative_z[..., beta_index]
        )
    return product


def spatial_operator(
    state: np.ndarray,
    dx: float,
    dz: float,
    gravity: float = GRAVITY_M_PER_S2,
) -> np.ndarray:
    flux_interfaces_x = _rusanov_fluxes(state, gravity, axis=0)
    flux_interfaces_z = _rusanov_fluxes(state, gravity, axis=1)
    divergence_x = (flux_interfaces_x[1:] - flux_interfaces_x[:-1]) / dx
    divergence_z = (flux_interfaces_z[:, 1:] - flux_interfaces_z[:, :-1]) / dz
    return (
        -divergence_x
        - divergence_z
        - nonconservative_product(state, dx, dz)
    )


def stable_timestep(
    state: np.ndarray,
    dx: float,
    dz: float,
    cfl: float,
    gravity: float = GRAVITY_M_PER_S2,
) -> float:
    speed_x = float(np.max(max_speed_x(state, gravity)))
    speed_z = float(np.max(max_speed_z(state, gravity)))
    return cfl / (speed_x / dx + speed_z / dz)


def _validated(state: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("the reduced-model state contains non-finite data")
    if float(np.min(state[..., 0])) <= 0.0:
        raise FloatingPointError("the reduced-model state lost positive depth")
    return state


def ssprk2_step(
    state: np.ndarray,
    dt: float,
    dx: float,
    dz: float,
    gravity: float = GRAVITY_M_PER_S2,
) -> np.ndarray:
    stage = _validated(state + dt * spatial_operator(state, dx, dz, gravity))
    return _validated(
        0.5 * state
        + 0.5 * (stage + dt * spatial_operator(stage, dx, dz, gravity))
    )


def state_diagnostics(
    state: np.ndarray,
    dx: float,
    dz: float,
    gravity: float,
) -> dict[str, float]:
    height, mean_u, mean_v, alpha, beta = primitive(state)
    kinetic = 0.5 * height * (mean_u**2 + mean_v**2)
    if alpha.shape[-1]:
        denominators = 2.0 * np.arange(1, alpha.shape[-1] + 1) + 1.0
        kinetic += 0.5 * height * np.sum(
            (alpha**2 + beta**2) / denominators, axis=-1
        )
        maximum_moment = float(max(np.max(np.abs(alpha)), np.max(np.abs(beta))))
    else:
        maximum_moment = 0.0
    energy_density = kinetic + 0.5 * gravity * height**2
    return {
        "water_volume_m3": float(np.sum(height, dtype=np.float64) * dx * dz),
        "x_momentum_m4_per_s": float(
            np.sum(state[..., 1], dtype=np.float64) * dx * dz
        ),
        "z_momentum_m4_per_s": float(
            np.sum(state[..., 2], dtype=np.float64) * dx * dz
        ),
        "mechanical_energy": float(
            np.sum(energy_density, dtype=np.float64) * dx * dz
        ),
        "minimum_depth_m": float(np.min(height)),
        "maximum_depth_m": float(np.max(height)),
        "maximum_speed_m_per_s": float(
            np.max(np.sqrt(mean_u**2 + mean_v**2))
        ),
        "maximum_absolute_moment_m_per_s": maximum_moment,
    }


def solve(
    nx: int,
    nz: int,
    order: int = 0,
    final_time: float = 3.0,
    cfl: float = 0.4,
    gravity: float = GRAVITY_M_PER_S2,
    output_times: tuple[float, ...] = OUTPUT_TIMES_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Advance the radial collapse and retain only requested snapshots."""

    _check_order(order)
    if final_time <= 0.0 or cfl <= 0.0:
        raise ValueError("final_time and cfl must be positive")
    x, z, state = initial_state(nx, nz, order)
    dx = DOMAIN_LENGTH_M / nx
    dz = DOMAIN_LENGTH_M / nz
    requested = tuple(
        value for value in output_times if 0.0 <= value <= final_time
    )
    if not requested or requested[0] != 0.0:
        raise ValueError("output_times must begin with 0.0")
    if requested[-1] != final_time:
        requested = requested + (final_time,)

    snapshots = [state.copy()]
    saved_times = [0.0]
    initial_diagnostics = state_diagnostics(state, dx, dz, gravity)
    initial_diagnostics["time_steps"] = 0.0
    diagnostics = [initial_diagnostics]
    time = 0.0
    steps = 0
    next_output = 1
    tolerance = 2.0e-13

    while time < final_time - tolerance:
        dt = stable_timestep(state, dx, dz, cfl, gravity)
        dt = min(dt, final_time - time)
        if next_output < len(requested):
            dt = min(dt, requested[next_output] - time)
        state = ssprk2_step(state, dt, dx, dz, gravity)
        time += dt
        steps += 1
        if next_output < len(requested) and math.isclose(
            time, requested[next_output], rel_tol=0.0, abs_tol=tolerance
        ):
            time = requested[next_output]
            snapshots.append(state.copy())
            saved_times.append(time)
            output_diagnostics = state_diagnostics(state, dx, dz, gravity)
            output_diagnostics["time_steps"] = float(steps)
            diagnostics.append(output_diagnostics)
            next_output += 1

    if next_output != len(requested):
        raise RuntimeError("the solver did not save every requested output time")
    return x, z, np.asarray(saved_times), np.stack(snapshots), diagnostics
