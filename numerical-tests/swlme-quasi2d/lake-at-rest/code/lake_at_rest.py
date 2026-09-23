"""Self-contained quasi-2D G-SWLME lake-at-rest discretization.

The test advances one representative x row of the complete quasi-two-
dimensional state

    U = (h, h*u_m, h*v_m, h*alpha_1, h*beta_1, ...)^T.

The configured problem is frictionless.  The only source is the bed-slope
term, and the analytic lake-at-rest state is prescribed directly instead of
using the singular moving-equilibrium construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss


Array = np.ndarray
METHOD_LABELS = {
    "standard": "standard first-order HLL",
    "wb1": "first-order well-balanced",
    "wb2": "second-order well-balanced",
}

PATH_NODES, PATH_WEIGHTS = leggauss(4)
PATH_NODES = 0.5 * (PATH_NODES + 1.0)
PATH_WEIGHTS = 0.5 * PATH_WEIGHTS


@dataclass(frozen=True)
class ModelParameters:
    """Dimensionless parameters used by this verification problem."""

    moments: int = 2
    gravity: float = 1.0
    free_surface: float = 1.0

    def __post_init__(self) -> None:
        if self.moments < 1:
            raise ValueError("At least one moment pair is required.")
        if self.gravity <= 0.0:
            raise ValueError("Gravity must be positive.")


@dataclass(frozen=True)
class SmoothBottom:
    """Smooth asymmetric multi-bump bottom used in the test."""

    offset: float = 0.15
    first_amplitude: float = 0.20
    first_center: float = 0.32
    first_width: float = 100.0
    second_amplitude: float = 0.10
    second_center: float = 0.70
    second_width: float = 200.0
    sine_amplitude: float = 0.03

    def value(self, x: float | Array) -> float | Array:
        x_array = np.asarray(x)
        first_bump = self.first_amplitude * np.exp(
            -self.first_width * (x_array - self.first_center) ** 2
        )
        second_bump = self.second_amplitude * np.exp(
            -self.second_width * (x_array - self.second_center) ** 2
        )
        result = (
            self.offset
            + first_bump
            + second_bump
            + self.sine_amplitude * np.sin(2.0 * np.pi * x_array)
        )
        return float(result) if np.isscalar(x) else result


@dataclass(frozen=True)
class Mesh:
    cells: int
    xmin: float = 0.0
    xmax: float = 1.0

    def __post_init__(self) -> None:
        if self.cells < 2:
            raise ValueError("The mesh must contain at least two cells.")
        if self.xmax <= self.xmin:
            raise ValueError("The mesh interval must have positive length.")

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / self.cells

    @property
    def centers(self) -> Array:
        return self.xmin + (np.arange(self.cells) + 0.5) * self.dx

    @property
    def interfaces(self) -> Array:
        return np.linspace(self.xmin, self.xmax, self.cells + 1)


@dataclass(frozen=True)
class LakeEquilibrium:
    """Cell, interface, and slope data for one discrete rest state."""

    cells: Array
    interfaces: Array
    slopes: Array
    bottom_cells: Array
    bottom_interfaces: Array
    bottom_slopes: Array


class Quasi2DGSWLME:
    """x-direction globally hyperbolic SWLME operator in vector-pair order."""

    def __init__(self, parameters: ModelParameters) -> None:
        self.p = parameters

    @property
    def size(self) -> int:
        return 3 + 2 * self.p.moments

    @staticmethod
    def alpha_index(moment: int) -> int:
        return 3 + 2 * moment

    @staticmethod
    def beta_index(moment: int) -> int:
        return 4 + 2 * moment

    def primitive_components(
        self, state: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        height = state[..., 0]
        if np.any(height <= 0.0):
            raise ValueError("The coefficient matrix requires positive water depth.")
        mean_u = state[..., 1] / height
        mean_v = state[..., 2] / height
        alpha = np.stack(
            [state[..., self.alpha_index(j)] / height for j in range(self.p.moments)],
            axis=-1,
        )
        beta = np.stack(
            [state[..., self.beta_index(j)] / height for j in range(self.p.moments)],
            axis=-1,
        )
        return height, mean_u, mean_v, alpha, beta

    def coefficient_product(self, state: Array, vector: Array) -> Array:
        """Evaluate A_G(U) times a vector without hiding the matrix entries."""
        height, mean_u, mean_v, alpha, beta = self.primitive_components(state)
        result = np.zeros_like(vector, dtype=float)
        result[..., 0] = vector[..., 1]

        weighted_alpha_square = np.zeros_like(height, dtype=float)
        weighted_alpha_beta = np.zeros_like(height, dtype=float)
        for j in range(self.p.moments):
            denominator = 2 * (j + 1) + 1
            weighted_alpha_square += alpha[..., j] ** 2 / denominator
            weighted_alpha_beta += alpha[..., j] * beta[..., j] / denominator

        result[..., 1] = (
            (self.p.gravity * height - mean_u**2 - weighted_alpha_square)
            * vector[..., 0]
            + 2.0 * mean_u * vector[..., 1]
        )
        result[..., 2] = (
            (-mean_u * mean_v - weighted_alpha_beta) * vector[..., 0]
            + mean_v * vector[..., 1]
            + mean_u * vector[..., 2]
        )

        for j in range(self.p.moments):
            denominator = 2 * (j + 1) + 1
            alpha_index = self.alpha_index(j)
            beta_index = self.beta_index(j)
            result[..., 1] += (
                2.0 * alpha[..., j] * vector[..., alpha_index] / denominator
            )
            result[..., 2] += (
                2.0 * alpha[..., j] * vector[..., beta_index] / denominator
            )
            result[..., alpha_index] = (
                -2.0 * mean_u * alpha[..., j] * vector[..., 0]
                + 2.0 * alpha[..., j] * vector[..., 1]
                + mean_u * vector[..., alpha_index]
            )
            result[..., beta_index] = (
                -(mean_u * beta[..., j] + mean_v * alpha[..., j])
                * vector[..., 0]
                + beta[..., j] * vector[..., 1]
                + alpha[..., j] * vector[..., 2]
                + mean_u * vector[..., beta_index]
            )
        return result

    def coefficient_matrix(self, state: Array) -> Array:
        """Return A_G(U), primarily for focused verification checks."""
        if state.ndim != 1:
            raise ValueError("coefficient_matrix expects one state vector.")
        columns = []
        for column in range(self.size):
            basis_vector = np.zeros(self.size)
            basis_vector[column] = 1.0
            columns.append(self.coefficient_product(state, basis_vector))
        return np.column_stack(columns)

    def topography_vector(self, state: Array) -> Array:
        result = np.zeros_like(state, dtype=float)
        result[..., 1] = self.p.gravity * state[..., 0]
        return result

    def characteristic_bounds(self, state: Array) -> tuple[Array, Array]:
        height, mean_u, _, alpha, _ = self.primitive_components(state)
        weighted_alpha_square = np.zeros_like(height, dtype=float)
        for j in range(self.p.moments):
            denominator = 2 * (j + 1) + 1
            weighted_alpha_square += alpha[..., j] ** 2 / denominator
        acoustic_speed = np.sqrt(
            self.p.gravity * height + 3.0 * weighted_alpha_square
        )
        return mean_u - acoustic_speed, mean_u + acoustic_speed

    def maximum_speed(self, state: Array) -> float:
        left, right = self.characteristic_bounds(state)
        return float(np.max(np.maximum(np.abs(left), np.abs(right))))


def build_lake_equilibrium(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    bottom: SmoothBottom,
) -> LakeEquilibrium:
    """Prescribe rest data for the continuous piecewise-linear discrete bed."""
    bottom_interfaces = np.asarray(bottom.value(mesh.interfaces), dtype=float)
    bottom_cells = 0.5 * (bottom_interfaces[:-1] + bottom_interfaces[1:])
    bottom_slopes = np.diff(bottom_interfaces) / mesh.dx

    height_interfaces = model.p.free_surface - bottom_interfaces
    height_cells = model.p.free_surface - bottom_cells
    if np.min(height_interfaces) <= 0.0 or np.min(height_cells) <= 0.0:
        raise ValueError("The selected free surface does not fully wet the bottom.")

    cells = np.zeros((mesh.cells, model.size))
    interfaces = np.zeros((mesh.cells + 1, model.size))
    slopes = np.zeros_like(cells)
    cells[:, 0] = height_cells
    interfaces[:, 0] = height_interfaces
    slopes[:, 0] = -bottom_slopes
    return LakeEquilibrium(
        cells=cells,
        interfaces=interfaces,
        slopes=slopes,
        bottom_cells=bottom_cells,
        bottom_interfaces=bottom_interfaces,
        bottom_slopes=bottom_slopes,
    )


def minmod(left: Array, right: Array) -> Array:
    result = np.zeros_like(left)
    same_sign = left * right > 0.0
    result[same_sign] = np.sign(left[same_sign]) * np.minimum(
        np.abs(left[same_sign]), np.abs(right[same_sign])
    )
    return result


def deviation_slopes(
    state: Array,
    equilibrium: LakeEquilibrium,
    mesh: Mesh,
    order: int,
) -> Array:
    if order == 1:
        return np.zeros_like(state)
    if order != 2:
        raise ValueError("The well-balanced spatial order must be one or two.")
    deviation = state - equilibrium.cells
    extended = np.zeros((mesh.cells + 2, state.shape[1]))
    extended[1:-1] = deviation
    backward = (extended[1:-1] - extended[:-2]) / mesh.dx
    forward = (extended[2:] - extended[1:-1]) / mesh.dx
    return minmod(backward, forward)


def path_integral(model: Quasi2DGSWLME, left: Array, right: Array) -> Array:
    jump = right - left
    result = np.zeros_like(jump)
    for node, weight in zip(PATH_NODES, PATH_WEIGHTS):
        path_state = left + node * jump
        result += weight * model.coefficient_product(path_state, jump)
    return result


def hll_fluctuations(
    model: Quasi2DGSWLME,
    left: Array,
    right: Array,
) -> tuple[Array, Array]:
    """Return path-conservative HLL fluctuations at one or many interfaces."""
    path_jump = path_integral(model, left, right)
    left_minimum, left_maximum = model.characteristic_bounds(left)
    right_minimum, right_maximum = model.characteristic_bounds(right)
    speed_left = np.minimum(0.0, np.minimum(left_minimum, right_minimum))
    speed_right = np.maximum(0.0, np.maximum(left_maximum, right_maximum))

    negative = np.zeros_like(path_jump)
    positive = np.zeros_like(path_jump)
    central = (speed_left < 0.0) & (speed_right > 0.0)
    if np.any(central):
        denominator = speed_right[central] - speed_left[central]
        star = (
            speed_right[central, None] * right[central]
            - speed_left[central, None] * left[central]
            - path_jump[central]
        ) / denominator[:, None]
        negative[central] = speed_left[central, None] * (star - left[central])
        positive[central] = speed_right[central, None] * (right[central] - star)

    right_going = speed_left >= 0.0
    positive[right_going] = path_jump[right_going]
    left_going = speed_right <= 0.0
    negative[left_going] = path_jump[left_going]
    return negative, positive


def well_balanced_operator(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: LakeEquilibrium,
    state: Array,
    order: int,
) -> Array:
    slopes = deviation_slopes(state, equilibrium, mesh, order)
    deviation = state - equilibrium.cells
    reconstructed_left = (
        equilibrium.interfaces[:-1] + deviation - 0.5 * mesh.dx * slopes
    )
    reconstructed_right = (
        equilibrium.interfaces[1:] + deviation + 0.5 * mesh.dx * slopes
    )

    interface_left = np.vstack((equilibrium.interfaces[0], reconstructed_right))
    interface_right = np.vstack((reconstructed_left, equilibrium.interfaces[-1]))
    negative, positive = hll_fluctuations(model, interface_left, interface_right)

    current_volume = model.coefficient_product(
        state, equilibrium.slopes + slopes
    )
    equilibrium_volume = model.coefficient_product(
        equilibrium.cells, equilibrium.slopes
    )
    topography_difference = (
        model.topography_vector(state)
        - model.topography_vector(equilibrium.cells)
    ) * equilibrium.bottom_slopes[:, None]
    correction = mesh.dx * (
        current_volume - equilibrium_volume + topography_difference
    )
    return -(negative[1:] + positive[:-1] + correction) / mesh.dx


def standard_operator(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: LakeEquilibrium,
    state: Array,
) -> Array:
    """First-order path-conservative HLL control without equilibrium subtraction."""
    interface_left = np.vstack((equilibrium.interfaces[0], state))
    interface_right = np.vstack((state, equilibrium.interfaces[-1]))
    negative, positive = hll_fluctuations(model, interface_left, interface_right)
    topography = model.topography_vector(state) * equilibrium.bottom_slopes[:, None]
    return -(negative[1:] + positive[:-1]) / mesh.dx - topography


def spatial_operator(
    method: str,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: LakeEquilibrium,
    state: Array,
) -> Array:
    if method == "standard":
        return standard_operator(model, mesh, equilibrium, state)
    if method == "wb1":
        return well_balanced_operator(model, mesh, equilibrium, state, order=1)
    if method == "wb2":
        return well_balanced_operator(model, mesh, equilibrium, state, order=2)
    raise ValueError(f"Unknown method {method!r}.")


def stable_time_step(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    state: Array,
    cfl: float,
) -> float:
    if not 0.0 < cfl <= 1.0:
        raise ValueError("The CFL number must lie in (0, 1].")
    return cfl * mesh.dx / max(model.maximum_speed(state), 1.0e-14)


def advance(
    method: str,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: LakeEquilibrium,
    state: Array,
    time_step: float,
) -> Array:
    first_rhs = spatial_operator(method, model, mesh, equilibrium, state)
    if method in ("standard", "wb1"):
        result = state + time_step * first_rhs
    elif method == "wb2":
        stage = state + time_step * first_rhs
        second_rhs = spatial_operator(method, model, mesh, equilibrium, stage)
        result = 0.5 * state + 0.5 * (stage + time_step * second_rhs)
    else:
        raise ValueError(f"Unknown method {method!r}.")
    if not np.isfinite(result).all() or np.min(result[:, 0]) <= 0.0:
        raise RuntimeError("The update produced a non-finite or non-positive state.")
    return result


def evolve_to_times(
    method: str,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: LakeEquilibrium,
    initial_state: Array,
    output_times: Iterable[float],
    cfl: float = 0.25,
) -> tuple[dict[float, Array], dict[float, int]]:
    targets = sorted(set(float(value) for value in output_times))
    if not targets or targets[0] < 0.0:
        raise ValueError("Output times must be a nonempty set of nonnegative values.")
    state = initial_state.copy()
    time = 0.0
    steps = 0
    snapshots: dict[float, Array] = {}
    step_counts: dict[float, int] = {}
    for target in targets:
        while time < target - 10.0 * np.finfo(float).eps:
            time_step = min(stable_time_step(model, mesh, state, cfl), target - time)
            state = advance(
                method, model, mesh, equilibrium, state, time_step
            )
            time += time_step
            steps += 1
        snapshots[target] = state.copy()
        step_counts[target] = steps
    return snapshots, step_counts


def primitive_array(model: Quasi2DGSWLME, state: Array) -> Array:
    height, mean_u, mean_v, alpha, beta = model.primitive_components(state)
    columns = [height, mean_u, mean_v]
    for j in range(model.p.moments):
        columns.extend((alpha[..., j], beta[..., j]))
    return np.stack(columns, axis=-1)


def diagnostic_values(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: LakeEquilibrium,
    state: Array,
) -> dict[str, float]:
    primitive = primitive_array(model, state)
    free_surface_error = state[:, 0] + equilibrium.bottom_cells - model.p.free_surface
    moment_maximum = float(np.max(np.abs(primitive[:, 3:])))
    initial_mass = mesh.dx * float(np.sum(equilibrium.cells[:, 0]))
    current_mass = mesh.dx * float(np.sum(state[:, 0]))
    return {
        "free_surface_l1": mesh.dx * float(np.sum(np.abs(free_surface_error))),
        "free_surface_linf": float(np.max(np.abs(free_surface_error))),
        "mean_u_linf": float(np.max(np.abs(primitive[:, 1]))),
        "mean_v_linf": float(np.max(np.abs(primitive[:, 2]))),
        "moments_linf": moment_maximum,
        "state_drift_linf": float(np.max(np.abs(state - equilibrium.cells))),
        "mass_error": abs(current_mass - initial_mass),
        "minimum_depth": float(np.min(state[:, 0])),
    }


def residual_values(
    method: str,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: LakeEquilibrium,
) -> dict[str, float]:
    residual = spatial_operator(
        method, model, mesh, equilibrium, equilibrium.cells.copy()
    )
    return {
        "residual_h_linf": float(np.max(np.abs(residual[:, 0]))),
        "residual_momentum_linf": float(np.max(np.abs(residual[:, 1:3]))),
        "residual_moments_linf": float(np.max(np.abs(residual[:, 3:]))),
        "residual_full_linf": float(np.max(np.abs(residual))),
    }
