"""Independent quasi-two-dimensional G-SWLME solver for Example 4.

The state is

    U = (h, h*u_m, h*v_m, h*alpha_1, h*beta_1, ...)^T.

All fields depend on x only, but both horizontal velocity components and both
moment families remain active.  The source is the standard projected
Navier-slip source used by Example 3B.  No effective wall coefficient or
source-term modification is present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Callable, Iterable

import numpy as np
from numpy.linalg import LinAlgError, norm, solve
from numpy.polynomial.legendre import Legendre, leggauss


Array = np.ndarray
METHOD_LABELS = {
    "standard": "standard first-order non-WB HLL",
    "wb1": "first-order well-balanced",
    "wb2": "second-order well-balanced",
}

PATH_NODES, PATH_WEIGHTS = leggauss(4)
PATH_NODES = 0.5 * (PATH_NODES + 1.0)
PATH_WEIGHTS = 0.5 * PATH_WEIGHTS


@dataclass(frozen=True)
class Parameters:
    """Dimensionless G-SWLME and standard-source parameters."""

    moments: int = 2
    gravity: float = 1.0
    epsilon: float = 0.1
    gamma: float = 0.002
    inverse_reynolds: float = 0.0005

    def __post_init__(self) -> None:
        if self.moments < 1:
            raise ValueError("At least one moment is required.")
        if self.gravity <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("Gravity and epsilon must be positive.")
        if self.gamma < 0.0 or self.inverse_reynolds < 0.0:
            raise ValueError("Friction parameters must be nonnegative.")

    def metadata(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class GaussianBottom:
    """Gaussian bottom shared with Example 3."""

    amplitude: float = 0.1
    center: float = 0.5
    width: float = 0.15

    def value(self, x: float | Array) -> float | Array:
        x_array = np.asarray(x)
        result = self.amplitude * np.exp(
            -((x_array - self.center) / self.width) ** 2
        )
        return float(result) if np.isscalar(x) else result

    def derivative(self, x: float | Array) -> float | Array:
        x_array = np.asarray(x)
        gaussian = self.amplitude * np.exp(
            -((x_array - self.center) / self.width) ** 2
        )
        result = -2.0 * (x_array - self.center) * gaussian / self.width**2
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


@dataclass
class DiscreteEquilibrium:
    cells: Array
    interfaces: Array
    slopes: Array
    newton_iterations: Array
    newton_residuals: Array


class Quasi2DGSWLME:
    """Quasi-2D globally hyperbolic SWLME with the standard source."""

    def __init__(self, parameters: Parameters, bottom: GaussianBottom) -> None:
        self.p = parameters
        self.bottom = bottom
        self.C = self._build_c_matrix(parameters.moments)

    @property
    def size(self) -> int:
        return 3 + 2 * self.p.moments

    @staticmethod
    def alpha_index(moment_index: int) -> int:
        return 3 + 2 * moment_index

    @staticmethod
    def beta_index(moment_index: int) -> int:
        return 4 + 2 * moment_index

    @staticmethod
    def _basis_derivative(index: int, zeta: Array) -> Array:
        polynomial = Legendre.basis(index).deriv()
        return -2.0 * polynomial(1.0 - 2.0 * zeta)

    @classmethod
    def _build_c_matrix(cls, moments: int, order: int = 40) -> Array:
        nodes, weights = leggauss(order)
        zeta = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        result = np.zeros((moments, moments))
        for i in range(1, moments + 1):
            derivative_i = cls._basis_derivative(i, zeta)
            for j in range(1, moments + 1):
                derivative_j = cls._basis_derivative(j, zeta)
                result[i - 1, j - 1] = np.sum(
                    weights * derivative_i * derivative_j
                )
        result[np.abs(result) < 5.0e-13] = 0.0
        return result

    def primitive_components(
        self, state: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        state_array = np.asarray(state, dtype=float)
        height = state_array[..., 0]
        if np.any(height <= 0.0):
            raise ValueError("The state contains non-positive water height.")
        mean_u = state_array[..., 1] / height
        mean_v = state_array[..., 2] / height
        alpha = np.stack(
            [
                state_array[..., self.alpha_index(j)] / height
                for j in range(self.p.moments)
            ],
            axis=-1,
        )
        beta = np.stack(
            [
                state_array[..., self.beta_index(j)] / height
                for j in range(self.p.moments)
            ],
            axis=-1,
        )
        return height, mean_u, mean_v, alpha, beta

    def coefficient_product(self, state: Array, vector: Array) -> Array:
        """Evaluate A_G(U) times a vector."""
        state_array = np.asarray(state, dtype=float)
        vector_array = np.asarray(vector, dtype=float)
        height, mean_u, mean_v, alpha, beta = self.primitive_components(
            state_array
        )
        result = np.zeros_like(vector_array, dtype=float)
        result[..., 0] = vector_array[..., 1]

        weighted_alpha_square = np.zeros_like(height, dtype=float)
        weighted_alpha_beta = np.zeros_like(height, dtype=float)
        for j in range(self.p.moments):
            denominator = 2 * (j + 1) + 1
            weighted_alpha_square += alpha[..., j] ** 2 / denominator
            weighted_alpha_beta += alpha[..., j] * beta[..., j] / denominator

        result[..., 1] = (
            (self.p.gravity * height - mean_u**2 - weighted_alpha_square)
            * vector_array[..., 0]
            + 2.0 * mean_u * vector_array[..., 1]
        )
        result[..., 2] = (
            (-mean_u * mean_v - weighted_alpha_beta) * vector_array[..., 0]
            + mean_v * vector_array[..., 1]
            + mean_u * vector_array[..., 2]
        )

        for j in range(self.p.moments):
            denominator = 2 * (j + 1) + 1
            alpha_index = self.alpha_index(j)
            beta_index = self.beta_index(j)
            result[..., 1] += (
                2.0
                * alpha[..., j]
                * vector_array[..., alpha_index]
                / denominator
            )
            result[..., 2] += (
                2.0
                * alpha[..., j]
                * vector_array[..., beta_index]
                / denominator
            )
            result[..., alpha_index] = (
                -2.0 * mean_u * alpha[..., j] * vector_array[..., 0]
                + 2.0 * alpha[..., j] * vector_array[..., 1]
                + mean_u * vector_array[..., alpha_index]
            )
            result[..., beta_index] = (
                -(mean_u * beta[..., j] + mean_v * alpha[..., j])
                * vector_array[..., 0]
                + beta[..., j] * vector_array[..., 1]
                + alpha[..., j] * vector_array[..., 2]
                + mean_u * vector_array[..., beta_index]
            )
        return result

    def coefficient_matrix(self, state: Array) -> Array:
        if np.asarray(state).ndim != 1:
            raise ValueError("coefficient_matrix expects one state vector.")
        columns = []
        for column in range(self.size):
            basis_vector = np.zeros(self.size)
            basis_vector[column] = 1.0
            columns.append(self.coefficient_product(state, basis_vector))
        return np.column_stack(columns)

    def topography_vector(self, state: Array) -> Array:
        state_array = np.asarray(state, dtype=float)
        result = np.zeros_like(state_array)
        result[..., 1] = self.p.gravity * state_array[..., 0]
        return result

    def standard_source(self, state: Array) -> Array:
        """Return the positive left-hand-side projected Navier-slip source."""
        state_array = np.asarray(state, dtype=float)
        height, mean_u, mean_v, alpha, beta = self.primitive_components(
            state_array
        )
        result = np.zeros_like(state_array)
        wall_rate = self.p.gamma / self.p.epsilon
        bottom_u = mean_u + np.sum(alpha, axis=-1)
        bottom_v = mean_v + np.sum(beta, axis=-1)
        result[..., 1] = wall_rate * bottom_u
        result[..., 2] = wall_rate * bottom_v

        viscous_rate = self.p.inverse_reynolds / (self.p.epsilon * height)
        for i in range(self.p.moments):
            coefficient = 2 * (i + 1) + 1
            alpha_relaxation = np.sum(alpha * self.C[i], axis=-1)
            beta_relaxation = np.sum(beta * self.C[i], axis=-1)
            result[..., self.alpha_index(i)] = coefficient * (
                wall_rate * bottom_u + viscous_rate * alpha_relaxation
            )
            result[..., self.beta_index(i)] = coefficient * (
                wall_rate * bottom_v + viscous_rate * beta_relaxation
            )
        return result

    def stationary_slope(self, state: Array, x: float) -> Array:
        forcing = (
            self.topography_vector(state) * self.bottom.derivative(x)
            + self.standard_source(state)
        )
        return -solve(self.coefficient_matrix(state), forcing)

    def characteristic_bounds(self, state: Array) -> tuple[Array, Array]:
        height, mean_u, _, alpha, _ = self.primitive_components(state)
        weighted_alpha_square = np.zeros_like(height, dtype=float)
        for j in range(self.p.moments):
            weighted_alpha_square += alpha[..., j] ** 2 / (2 * (j + 1) + 1)
        acoustic_speed = np.sqrt(
            self.p.gravity * height + 3.0 * weighted_alpha_square
        )
        return mean_u - acoustic_speed, mean_u + acoustic_speed

    def maximum_speed(self, state: Array) -> float:
        left, right = self.characteristic_bounds(state)
        return float(np.max(np.maximum(np.abs(left), np.abs(right))))

    def source_time_step(self, state: Array, source_cfl: float = 0.5) -> float:
        """Return a conservative explicit step based on source scales."""
        height = np.asarray(state)[..., 0]
        minimum_height = float(np.min(height))
        wall_rate = self.p.gamma / (self.p.epsilon * minimum_height)
        viscous_rate = self.p.inverse_reynolds / (
            self.p.epsilon * minimum_height**2
        )
        maximum_row_sum = wall_rate * (self.p.moments + 1)
        for i in range(self.p.moments):
            coefficient = 2 * (i + 1) + 1
            row_sum = coefficient * (
                wall_rate * (self.p.moments + 1)
                + viscous_rate * np.sum(np.abs(self.C[i]))
            )
            maximum_row_sum = max(maximum_row_sum, float(row_sum))
        return source_cfl / maximum_row_sum


def conservative_state(
    height: float,
    mean_u: float,
    mean_v: float,
    alpha: Iterable[float],
    beta: Iterable[float],
) -> Array:
    alpha_values = list(alpha)
    beta_values = list(beta)
    if len(alpha_values) != len(beta_values):
        raise ValueError("The alpha and beta families must have the same length.")
    state = np.zeros(3 + 2 * len(alpha_values))
    state[0] = height
    state[1] = height * mean_u
    state[2] = height * mean_v
    for j, (alpha_j, beta_j) in enumerate(zip(alpha_values, beta_values)):
        state[3 + 2 * j] = height * alpha_j
        state[4 + 2 * j] = height * beta_j
    return state


def numerical_jacobian(function: Callable[[Array], Array], state: Array) -> Array:
    result = np.zeros((state.size, state.size))
    step_scale = np.cbrt(np.finfo(float).eps)
    for j in range(state.size):
        step = step_scale * max(1.0, abs(state[j]))
        direction = np.zeros(state.size)
        direction[j] = step
        result[:, j] = (
            function(state + direction) - function(state - direction)
        ) / (2.0 * step)
    return result


def damped_newton(
    residual: Callable[[Array], Array],
    initial_guess: Array,
    tolerance: float = 5.0e-13,
    maximum_iterations: int = 40,
) -> tuple[Array, int, float]:
    state = initial_guess.copy()
    for iteration in range(1, maximum_iterations + 1):
        value = residual(state)
        value_norm = float(norm(value, ord=np.inf))
        if value_norm <= tolerance:
            return state, iteration - 1, value_norm
        try:
            correction = solve(numerical_jacobian(residual, state), -value)
        except LinAlgError as exc:
            raise RuntimeError("Singular equilibrium Newton Jacobian.") from exc
        damping = 1.0
        while damping >= 2.0**-30:
            candidate = state + damping * correction
            if candidate[0] > 1.0e-12 and np.isfinite(candidate).all():
                candidate_norm = float(norm(residual(candidate), ord=np.inf))
                if candidate_norm < value_norm:
                    state = candidate
                    break
            damping *= 0.5
        else:
            raise RuntimeError("Damped Newton iteration failed.")
    raise RuntimeError("Equilibrium Newton iteration did not converge.")


def build_equilibrium(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    left_state: Array,
    newton_tolerance: float = 5.0e-13,
) -> DiscreteEquilibrium:
    """Propagate the midpoint-collocation equilibrium from the left."""
    cells = np.zeros((mesh.cells, model.size))
    interfaces = np.zeros((mesh.cells + 1, model.size))
    slopes = np.zeros_like(cells)
    iterations = np.zeros(mesh.cells, dtype=int)
    residuals = np.zeros(mesh.cells)
    interfaces[0] = left_state
    guess = left_state.copy()

    for i, x in enumerate(mesh.centers):
        left_interface = interfaces[i].copy()

        def residual(candidate: Array) -> Array:
            return (
                candidate
                - 0.5 * mesh.dx * model.stationary_slope(candidate, float(x))
                - left_interface
            )

        cells[i], iterations[i], residuals[i] = damped_newton(
            residual, guess, tolerance=newton_tolerance
        )
        slopes[i] = model.stationary_slope(cells[i], float(x))
        interfaces[i + 1] = cells[i] + 0.5 * mesh.dx * slopes[i]
        guess = interfaces[i + 1]
    return DiscreteEquilibrium(
        cells=cells,
        interfaces=interfaces,
        slopes=slopes,
        newton_iterations=iterations,
        newton_residuals=residuals,
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
    equilibrium: DiscreteEquilibrium,
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
        result += weight * model.coefficient_product(left + node * jump, jump)
    return result


def hll_fluctuations(
    model: Quasi2DGSWLME, left: Array, right: Array
) -> tuple[Array, Array]:
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
        positive[central] = speed_right[central, None] * (
            right[central] - star
        )

    right_going = speed_left >= 0.0
    positive[right_going] = path_jump[right_going]
    left_going = speed_right <= 0.0
    negative[left_going] = path_jump[left_going]
    return negative, positive


def interface_fluctuations(
    model: Quasi2DGSWLME,
    left_traces: Array,
    right_traces: Array,
    left_boundary: Array,
    right_boundary: Array,
) -> tuple[Array, Array]:
    left_states = np.vstack((left_boundary, right_traces))
    right_states = np.vstack((left_traces, right_boundary))
    return hll_fluctuations(model, left_states, right_states)


def well_balanced_operator(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: DiscreteEquilibrium,
    state: Array,
    order: int,
) -> Array:
    slopes = deviation_slopes(state, equilibrium, mesh, order)
    deviation = state - equilibrium.cells
    left_traces = equilibrium.interfaces[:-1] + deviation - 0.5 * mesh.dx * slopes
    right_traces = equilibrium.interfaces[1:] + deviation + 0.5 * mesh.dx * slopes
    negative, positive = interface_fluctuations(
        model,
        left_traces,
        right_traces,
        equilibrium.interfaces[0],
        equilibrium.interfaces[-1],
    )

    correction = mesh.dx * (
        model.coefficient_product(state, equilibrium.slopes + slopes)
        - model.coefficient_product(equilibrium.cells, equilibrium.slopes)
        + (
            model.topography_vector(state)
            - model.topography_vector(equilibrium.cells)
        )
        * model.bottom.derivative(mesh.centers)[:, None]
        + model.standard_source(state)
        - model.standard_source(equilibrium.cells)
    )
    return -(negative[1:] + positive[:-1] + correction) / mesh.dx


def standard_operator(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: DiscreteEquilibrium,
    state: Array,
) -> Array:
    negative, positive = interface_fluctuations(
        model,
        state,
        state,
        equilibrium.interfaces[0],
        equilibrium.interfaces[-1],
    )
    source = (
        model.topography_vector(state)
        * model.bottom.derivative(mesh.centers)[:, None]
        + model.standard_source(state)
    )
    return -(negative[1:] + positive[:-1]) / mesh.dx - source


def spatial_operator(
    method: str,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: DiscreteEquilibrium,
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
    source_cfl: float,
) -> float:
    hyperbolic_step = cfl * mesh.dx / max(model.maximum_speed(state), 1.0e-14)
    return min(hyperbolic_step, model.source_time_step(state, source_cfl))


def advance(
    method: str,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: DiscreteEquilibrium,
    state: Array,
    time_step: float,
) -> Array:
    right_hand_side = spatial_operator(method, model, mesh, equilibrium, state)
    if method in ("standard", "wb1"):
        result = state + time_step * right_hand_side
    elif method == "wb2":
        stage = state + time_step * right_hand_side
        result = 0.5 * state + 0.5 * (
            stage
            + time_step
            * spatial_operator(method, model, mesh, equilibrium, stage)
        )
    else:
        raise ValueError(f"Unknown method {method!r}.")
    if not np.isfinite(result).all() or np.min(result[:, 0]) <= 0.0:
        raise RuntimeError("The update produced an invalid state.")
    return result


def evolve_to_times(
    method: str,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: DiscreteEquilibrium,
    initial_state: Array,
    output_times: Iterable[float],
    cfl: float = 0.25,
    source_cfl: float = 0.5,
) -> tuple[dict[float, Array], dict[float, int]]:
    targets = sorted(set(float(value) for value in output_times))
    if not targets or targets[0] < 0.0:
        raise ValueError("Output times must be nonnegative and nonempty.")
    state = initial_state.copy()
    time = 0.0
    steps = 0
    snapshots: dict[float, Array] = {}
    step_counts: dict[float, int] = {}
    for target in targets:
        while time < target - 10.0 * np.finfo(float).eps:
            time_step = min(
                stable_time_step(model, mesh, state, cfl, source_cfl),
                target - time,
            )
            state = advance(method, model, mesh, equilibrium, state, time_step)
            time += time_step
            steps += 1
        snapshots[target] = state.copy()
        step_counts[target] = steps
    return snapshots, step_counts


def primitive_names(moments: int) -> list[str]:
    names = ["h", "u_m", "v_m"]
    for index in range(1, moments + 1):
        names.extend([f"alpha_{index}", f"beta_{index}"])
    return names


def primitive_array(state: Array, moments: int) -> Array:
    height = state[:, 0]
    columns = [height, state[:, 1] / height, state[:, 2] / height]
    for j in range(moments):
        columns.extend(
            [
                state[:, 3 + 2 * j] / height,
                state[:, 4 + 2 * j] / height,
            ]
        )
    return np.column_stack(columns)


def component_errors(
    state: Array, reference: Array, dx: float, moments: int
) -> tuple[Array, Array]:
    difference = np.abs(
        primitive_array(state, moments) - primitive_array(reference, moments)
    )
    return dx * np.sum(difference, axis=0), np.max(difference, axis=0)


def block_average(fine_state: Array, coarse_cells: int) -> Array:
    if fine_state.shape[0] % coarse_cells != 0:
        raise ValueError("The fine grid must be divisible by the coarse grid.")
    ratio = fine_state.shape[0] // coarse_cells
    return fine_state.reshape(coarse_cells, ratio, fine_state.shape[1]).mean(axis=1)


def state_checksum(state: Array) -> str:
    contiguous = np.ascontiguousarray(state, dtype=np.float64)
    return sha256(contiguous.tobytes()).hexdigest()
