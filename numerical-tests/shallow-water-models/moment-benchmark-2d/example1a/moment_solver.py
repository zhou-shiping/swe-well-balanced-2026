"""Smooth periodic solver for the two-dimensional G-SWLME.

The state ordering is

    h, h*u_m, h*v_m, h*alpha_1, h*beta_1, ...

The implementation is intentionally self-contained.  It uses MUSCL
reconstruction, local Lax--Friedrichs fluxes for the conservative part,
centered products for the smooth nonconservative terms, and SSP-RK3.
The benchmark is stopped before shock formation.
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


def initial_state(nx: int, ny: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the cell-centered periodic initial state from the tracker."""
    if order not in (0, 1, 2):
        raise ValueError("the prepared benchmark supports moment orders N=0, 1, and 2")

    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")

    height = 1.0 + 0.05 * np.cos(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    mean_u = 0.2 + 0.02 * np.sin(2.0 * np.pi * yy)
    mean_v = 0.1 + 0.02 * np.sin(2.0 * np.pi * xx)

    state = np.zeros((nx, ny, 3 + 2 * order), dtype=np.float64)
    state[..., 0] = height
    state[..., 1] = height * mean_u
    state[..., 2] = height * mean_v

    if order >= 1:
        alpha_1 = 0.03 * (1.0 + 0.2 * np.cos(2.0 * np.pi * xx))
        beta_1 = 0.02 * (1.0 + 0.2 * np.sin(2.0 * np.pi * yy))
        state[..., 3] = height * alpha_1
        state[..., 4] = height * beta_1

    if order >= 2:
        alpha_2 = 0.01 * np.sin(2.0 * np.pi * (xx + yy))
        beta_2 = 0.015 * np.cos(2.0 * np.pi * (xx - yy))
        state[..., 5] = height * alpha_2
        state[..., 6] = height * beta_2

    return x, y, state


def _primitive(
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = state[..., 0]
    if np.any(height <= 0.0):
        raise FloatingPointError("the G-SWLME state contains non-positive water depth")
    mean_u = state[..., 1] / height
    mean_v = state[..., 2] / height
    alpha = state[..., 3::2] / height[..., None]
    beta = state[..., 4::2] / height[..., None]
    return height, mean_u, mean_v, alpha, beta


def flux_x(state: np.ndarray, gravity: float) -> np.ndarray:
    height, mean_u, mean_v, alpha, beta = _primitive(state)
    flux = np.zeros_like(state)
    flux[..., 0] = state[..., 1]

    if alpha.shape[-1]:
        denominators = 2.0 * np.arange(1, alpha.shape[-1] + 1) + 1.0
        alpha_energy = np.sum(alpha * alpha / denominators, axis=-1)
        cross_energy = np.sum(alpha * beta / denominators, axis=-1)
    else:
        alpha_energy = 0.0
        cross_energy = 0.0

    flux[..., 1] = height * (mean_u * mean_u + alpha_energy) + 0.5 * gravity * height**2
    flux[..., 2] = height * (mean_u * mean_v + cross_energy)

    for moment in range(alpha.shape[-1]):
        alpha_index = 3 + 2 * moment
        beta_index = alpha_index + 1
        flux[..., alpha_index] = 2.0 * height * mean_u * alpha[..., moment]
        flux[..., beta_index] = height * (
            mean_u * beta[..., moment] + mean_v * alpha[..., moment]
        )
    return flux


def flux_y(state: np.ndarray, gravity: float) -> np.ndarray:
    height, mean_u, mean_v, alpha, beta = _primitive(state)
    flux = np.zeros_like(state)
    flux[..., 0] = state[..., 2]

    if beta.shape[-1]:
        denominators = 2.0 * np.arange(1, beta.shape[-1] + 1) + 1.0
        beta_energy = np.sum(beta * beta / denominators, axis=-1)
        cross_energy = np.sum(alpha * beta / denominators, axis=-1)
    else:
        beta_energy = 0.0
        cross_energy = 0.0

    flux[..., 1] = height * (mean_u * mean_v + cross_energy)
    flux[..., 2] = height * (mean_v * mean_v + beta_energy) + 0.5 * gravity * height**2

    for moment in range(beta.shape[-1]):
        alpha_index = 3 + 2 * moment
        beta_index = alpha_index + 1
        flux[..., alpha_index] = height * (
            mean_u * beta[..., moment] + mean_v * alpha[..., moment]
        )
        flux[..., beta_index] = 2.0 * height * mean_v * beta[..., moment]
    return flux


def max_speed_x(state: np.ndarray, gravity: float) -> np.ndarray:
    height, mean_u, _, alpha, _ = _primitive(state)
    if alpha.shape[-1]:
        denominators = 2.0 * np.arange(1, alpha.shape[-1] + 1) + 1.0
        moment_energy = np.sum(alpha * alpha / denominators, axis=-1)
    else:
        moment_energy = 0.0
    return np.abs(mean_u) + np.sqrt(gravity * height + 3.0 * moment_energy)


def max_speed_y(state: np.ndarray, gravity: float) -> np.ndarray:
    height, _, mean_v, _, beta = _primitive(state)
    if beta.shape[-1]:
        denominators = 2.0 * np.arange(1, beta.shape[-1] + 1) + 1.0
        moment_energy = np.sum(beta * beta / denominators, axis=-1)
    else:
        moment_energy = 0.0
    return np.abs(mean_v) + np.sqrt(gravity * height + 3.0 * moment_energy)


def _rusanov_flux(state: np.ndarray, gravity: float, axis: int) -> np.ndarray:
    difference = _mc_difference(state, axis)
    left = state + 0.5 * difference
    right = np.roll(state, -1, axis=axis) - 0.5 * np.roll(difference, -1, axis=axis)
    if np.any(left[..., 0] <= 0.0) or np.any(right[..., 0] <= 0.0):
        raise FloatingPointError("MUSCL reconstruction produced a non-positive interface depth")

    if axis == 0:
        physical_left = flux_x(left, gravity)
        physical_right = flux_x(right, gravity)
        speed = np.maximum(max_speed_x(left, gravity), max_speed_x(right, gravity))
    elif axis == 1:
        physical_left = flux_y(left, gravity)
        physical_right = flux_y(right, gravity)
        speed = np.maximum(max_speed_y(left, gravity), max_speed_y(right, gravity))
    else:
        raise ValueError("the horizontal direction must be axis 0 or 1")

    return 0.5 * (physical_left + physical_right) - 0.5 * speed[..., None] * (right - left)


def nonconservative_product(state: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return (P_1+Delta A) U_x + (Q_1+Delta B) U_y for G-SWLME."""
    _, mean_u, mean_v, alpha, beta = _primitive(state)
    derivative_x = (np.roll(state, -1, axis=0) - np.roll(state, 1, axis=0)) / (2.0 * dx)
    derivative_y = (np.roll(state, -1, axis=1) - np.roll(state, 1, axis=1)) / (2.0 * dy)
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
            beta[..., moment] * derivative_y[..., alpha_index]
            - alpha[..., moment] * derivative_y[..., beta_index]
        ) / denominator

        product[..., alpha_index] += (
            -mean_u * derivative_x[..., alpha_index]
            -mean_u * derivative_y[..., beta_index]
        )
        product[..., beta_index] += (
            -mean_v * derivative_x[..., alpha_index]
            -mean_v * derivative_y[..., beta_index]
        )
    return product


def spatial_operator(state: np.ndarray, gravity: float, dx: float, dy: float) -> np.ndarray:
    interface_flux_x = _rusanov_flux(state, gravity, axis=0)
    interface_flux_y = _rusanov_flux(state, gravity, axis=1)
    divergence_x = (interface_flux_x - np.roll(interface_flux_x, 1, axis=0)) / dx
    divergence_y = (interface_flux_y - np.roll(interface_flux_y, 1, axis=1)) / dy
    return -divergence_x - divergence_y - nonconservative_product(state, dx, dy)


def stable_timestep(state: np.ndarray, gravity: float, dx: float, dy: float, cfl: float) -> float:
    speed_x = float(np.max(max_speed_x(state, gravity)))
    speed_y = float(np.max(max_speed_y(state, gravity)))
    return cfl / (speed_x / dx + speed_y / dy)


def _validated(state: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("the G-SWLME state contains a non-finite value")
    if float(np.min(state[..., 0])) <= 0.0:
        raise FloatingPointError("the G-SWLME state lost positive water depth")
    return state


def ssprk3_step(state: np.ndarray, dt: float, gravity: float, dx: float, dy: float) -> np.ndarray:
    stage_1 = _validated(state + dt * spatial_operator(state, gravity, dx, dy))
    stage_2 = _validated(
        0.75 * state
        + 0.25 * (stage_1 + dt * spatial_operator(stage_1, gravity, dx, dy))
    )
    return _validated(
        state / 3.0
        + 2.0 * (stage_2 + dt * spatial_operator(stage_2, gravity, dx, dy)) / 3.0
    )


def solve(
    nx: int,
    ny: int,
    order: int,
    final_time: float = 0.1,
    gravity: float = 1.0,
    cfl: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    x, y, state = initial_state(nx, ny, order)
    initial_mass = float(np.mean(state[..., 0]))
    dx = 1.0 / nx
    dy = 1.0 / ny
    time = 0.0
    steps = 0

    while time < final_time:
        dt = min(stable_timestep(state, gravity, dx, dy, cfl), final_time - time)
        state = ssprk3_step(state, dt, gravity, dx, dy)
        time += dt
        steps += 1

    final_mass = float(np.mean(state[..., 0]))
    diagnostics: dict[str, float | int] = {
        "steps": steps,
        "final_time": time,
        "minimum_depth": float(np.min(state[..., 0])),
        "maximum_depth": float(np.max(state[..., 0])),
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "relative_mass_defect": abs(final_mass - initial_mass) / abs(initial_mass),
        "maximum_state_magnitude": float(np.max(np.abs(state))),
    }
    if not math.isclose(time, final_time, rel_tol=0.0, abs_tol=2.0e-14):
        raise RuntimeError("the time integrator did not land on the requested final time")
    return x, y, state, diagnostics


def swap_xy_components(state: np.ndarray) -> np.ndarray:
    """Rotate the grid and exchange every horizontal vector pair."""
    swapped = np.swapaxes(state, 0, 1).copy()
    swapped[..., [1, 2]] = swapped[..., [2, 1]]
    for alpha_index in range(3, state.shape[-1], 2):
        beta_index = alpha_index + 1
        swapped[..., [alpha_index, beta_index]] = swapped[..., [beta_index, alpha_index]]
    return swapped
