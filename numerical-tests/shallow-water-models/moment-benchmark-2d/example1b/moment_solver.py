"""Local G-SWLME solver used only by the self-contained Example 1B comparison."""

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
    return _minmod3(1.5 * backward, 0.5 * (backward + forward), 1.5 * forward)


def initial_state(nx: int, ny: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order not in (0, 1, 2):
        raise ValueError("the prepared comparison supports moment orders N=0, 1, and 2")
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    height = 1.0 + 0.05 * np.cos(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    mean_u = 0.2 + 0.02 * np.sin(2.0 * np.pi * yy)
    mean_v = 0.1 + 0.02 * np.sin(2.0 * np.pi * xx)
    state = np.zeros((nx, ny, 3 + 2 * order))
    state[..., 0] = height
    state[..., 1] = height * mean_u
    state[..., 2] = height * mean_v
    if order >= 1:
        state[..., 3] = height * 0.03 * (1.0 + 0.2 * np.cos(2.0 * np.pi * xx))
        state[..., 4] = height * 0.02 * (1.0 + 0.2 * np.sin(2.0 * np.pi * yy))
    if order >= 2:
        state[..., 5] = height * 0.01 * np.sin(2.0 * np.pi * (xx + yy))
        state[..., 6] = height * 0.015 * np.cos(2.0 * np.pi * (xx - yy))
    return x, y, state


def _primitive(
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = state[..., 0]
    if np.any(height <= 0.0):
        raise FloatingPointError("the G-SWLME state contains non-positive depth")
    return (
        height,
        state[..., 1] / height,
        state[..., 2] / height,
        state[..., 3::2] / height[..., None],
        state[..., 4::2] / height[..., None],
    )


def _physical_flux(state: np.ndarray, gravity: float, axis: int) -> np.ndarray:
    height, mean_u, mean_v, alpha, beta = _primitive(state)
    normal_mean = mean_u if axis == 0 else mean_v
    normal_moment = alpha if axis == 0 else beta
    flux = np.zeros_like(state)
    flux[..., 0] = state[..., 1 + axis]
    if alpha.shape[-1]:
        denominators = 2.0 * np.arange(1, alpha.shape[-1] + 1) + 1.0
        alpha_energy = np.sum(alpha * alpha / denominators, axis=-1)
        beta_energy = np.sum(beta * beta / denominators, axis=-1)
        cross_energy = np.sum(alpha * beta / denominators, axis=-1)
    else:
        alpha_energy = beta_energy = cross_energy = 0.0
    flux[..., 1] = height * (
        mean_u * normal_mean + (alpha_energy if axis == 0 else cross_energy)
    )
    flux[..., 2] = height * (
        mean_v * normal_mean + (cross_energy if axis == 0 else beta_energy)
    )
    flux[..., 1 + axis] += 0.5 * gravity * height**2
    for moment in range(alpha.shape[-1]):
        alpha_index = 3 + 2 * moment
        beta_index = alpha_index + 1
        if axis == 0:
            flux[..., alpha_index] = 2.0 * height * mean_u * alpha[..., moment]
            flux[..., beta_index] = height * (
                mean_u * beta[..., moment] + mean_v * alpha[..., moment]
            )
        else:
            flux[..., alpha_index] = height * (
                mean_u * beta[..., moment] + mean_v * alpha[..., moment]
            )
            flux[..., beta_index] = 2.0 * height * mean_v * beta[..., moment]
    return flux


def _max_speed(state: np.ndarray, gravity: float, axis: int) -> np.ndarray:
    height, mean_u, mean_v, alpha, beta = _primitive(state)
    normal_mean = mean_u if axis == 0 else mean_v
    normal_moment = alpha if axis == 0 else beta
    if normal_moment.shape[-1]:
        denominators = 2.0 * np.arange(1, normal_moment.shape[-1] + 1) + 1.0
        energy = np.sum(normal_moment * normal_moment / denominators, axis=-1)
    else:
        energy = 0.0
    return np.abs(normal_mean) + np.sqrt(gravity * height + 3.0 * energy)


def _rusanov_flux(state: np.ndarray, gravity: float, axis: int) -> np.ndarray:
    difference = _mc_difference(state, axis)
    left = state + 0.5 * difference
    right = np.roll(state, -1, axis=axis) - 0.5 * np.roll(difference, -1, axis=axis)
    if np.any(left[..., 0] <= 0.0) or np.any(right[..., 0] <= 0.0):
        raise FloatingPointError("MUSCL reconstruction produced a non-positive depth")
    speed = np.maximum(_max_speed(left, gravity, axis), _max_speed(right, gravity, axis))
    return 0.5 * (
        _physical_flux(left, gravity, axis) + _physical_flux(right, gravity, axis)
    ) - 0.5 * speed[..., None] * (right - left)


def _nonconservative_product(state: np.ndarray, dx: float, dy: float) -> np.ndarray:
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
            - mean_u * derivative_y[..., beta_index]
        )
        product[..., beta_index] += (
            -mean_v * derivative_x[..., alpha_index]
            - mean_v * derivative_y[..., beta_index]
        )
    return product


def spatial_operator(state: np.ndarray, gravity: float, dx: float, dy: float) -> np.ndarray:
    flux_x = _rusanov_flux(state, gravity, axis=0)
    flux_y = _rusanov_flux(state, gravity, axis=1)
    divergence_x = (flux_x - np.roll(flux_x, 1, axis=0)) / dx
    divergence_y = (flux_y - np.roll(flux_y, 1, axis=1)) / dy
    return -divergence_x - divergence_y - _nonconservative_product(state, dx, dy)


def _stable_timestep(
    state: np.ndarray,
    gravity: float,
    dx: float,
    dy: float,
    cfl: float,
) -> float:
    return cfl / (
        float(np.max(_max_speed(state, gravity, axis=0))) / dx
        + float(np.max(_max_speed(state, gravity, axis=1))) / dy
    )


def _validated(state: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(state)) or float(np.min(state[..., 0])) <= 0.0:
        raise FloatingPointError("the G-SWLME state is not admissible")
    return state


def _ssprk3_step(
    state: np.ndarray,
    dt: float,
    gravity: float,
    dx: float,
    dy: float,
) -> np.ndarray:
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
        dt = min(_stable_timestep(state, gravity, dx, dy, cfl), final_time - time)
        state = _ssprk3_step(state, dt, gravity, dx, dy)
        time += dt
        steps += 1
    final_mass = float(np.mean(state[..., 0]))
    diagnostics: dict[str, float | int] = {
        "steps": steps,
        "final_time": time,
        "minimum_depth": float(np.min(state[..., 0])),
        "relative_mass_defect": abs(final_mass - initial_mass) / abs(initial_mass),
    }
    if not math.isclose(time, final_time, rel_tol=0.0, abs_tol=2.0e-14):
        raise RuntimeError("the time integrator did not reach the requested final time")
    return x, y, state, diagnostics
