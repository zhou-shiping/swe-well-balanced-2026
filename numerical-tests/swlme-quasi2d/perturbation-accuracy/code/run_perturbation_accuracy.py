"""Run Example 4: perturbations of the Example 3B continuous equilibrium."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp

from quasi2d_wb import (
    METHOD_LABELS,
    GaussianBottom,
    Mesh,
    Parameters,
    Quasi2DGSWLME,
    block_average,
    build_equilibrium,
    component_errors,
    conservative_state,
    evolve_to_times,
    primitive_array,
    primitive_names,
    state_checksum,
)


CASE_ROOT = Path(__file__).resolve().parents[1]


def configured_problem() -> tuple[Quasi2DGSWLME, np.ndarray]:
    """Return exactly the full-state standard-source branch from Example 3B."""
    model = Quasi2DGSWLME(
        Parameters(
            moments=2,
            gravity=1.0,
            epsilon=0.1,
            gamma=0.002,
            inverse_reynolds=0.0005,
        ),
        GaussianBottom(amplitude=0.1, center=0.5, width=0.15),
    )
    left_state = conservative_state(
        height=1.0,
        mean_u=0.4,
        mean_v=0.15,
        alpha=[0.08, -0.03],
        beta=[-0.04, 0.05],
    )
    return model, left_state


def stationary_solution(
    model: Quasi2DGSWLME,
    left_state: np.ndarray,
    relative_tolerance: float,
    absolute_tolerance: float,
):
    """Integrate the continuous stationary ODE with dense DOP853 output."""
    solution = solve_ivp(
        lambda x, state: model.stationary_slope(state, float(x)),
        (0.0, 1.0),
        left_state,
        method="DOP853",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"The stationary reference failed: {solution.message}")
    return solution


def cell_average_initial_data(
    stationary,
    mesh: Mesh,
    amplitude: float,
    center: float,
    sigma: float,
    quadrature_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average one fixed continuous equilibrium and perturbation over cells."""
    nodes, weights = leggauss(quadrature_points)
    points = mesh.centers[:, None] + 0.5 * mesh.dx * nodes[None, :]
    equilibrium_values = stationary.sol(points.ravel()).T.reshape(
        mesh.cells, quadrature_points, -1
    )
    if not np.isfinite(equilibrium_values).all():
        raise RuntimeError("The continuous stationary data are not finite.")

    initial_values = equilibrium_values.copy()
    old_height = equilibrium_values[..., 0]
    pulse = amplitude * np.exp(
        -((points - center) ** 2) / (2.0 * sigma**2)
    )
    new_height = old_height + pulse
    initial_values[..., 0] = new_height
    initial_values[..., 1:] *= (new_height / old_height)[..., None]

    equilibrium_average = 0.5 * np.einsum(
        "q,iqk->ik", weights, equilibrium_values
    )
    initial_average = 0.5 * np.einsum("q,iqk->ik", weights, initial_values)
    return equilibrium_average, initial_average


def amplitude_tag(amplitude: float) -> str:
    return f"{amplitude:g}".replace("-", "m").replace(".", "p")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows were provided for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def signal_norms(
    reference: np.ndarray,
    continuous_equilibrium: np.ndarray,
    dx: float,
    moments: int,
) -> tuple[np.ndarray, np.ndarray]:
    perturbation = np.abs(
        primitive_array(reference, moments)
        - primitive_array(continuous_equilibrium, moments)
    )
    return dx * np.sum(perturbation, axis=0), np.max(perturbation, axis=0)


def add_error_columns(
    row: dict[str, object],
    state: np.ndarray,
    reference: np.ndarray,
    continuous_equilibrium: np.ndarray,
    dx: float,
    names: list[str],
    moments: int,
) -> None:
    full_l1, full_linf = component_errors(state, reference, dx, moments)
    signal_l1, signal_linf = signal_norms(
        reference, continuous_equilibrium, dx, moments
    )

    numerical_perturbation = (
        primitive_array(state, moments)
        - primitive_array(continuous_equilibrium, moments)
    )
    reference_perturbation = (
        primitive_array(reference, moments)
        - primitive_array(continuous_equilibrium, moments)
    )
    perturbation_difference = np.abs(
        numerical_perturbation - reference_perturbation
    )
    perturbation_l1 = dx * np.sum(perturbation_difference, axis=0)
    perturbation_linf = np.max(perturbation_difference, axis=0)

    for index, name in enumerate(names):
        row[f"full_L1_{name}"] = float(full_l1[index])
        row[f"full_Linf_{name}"] = float(full_linf[index])
        row[f"perturbation_L1_{name}"] = float(perturbation_l1[index])
        row[f"perturbation_Linf_{name}"] = float(perturbation_linf[index])
        row[f"signal_L1_{name}"] = float(signal_l1[index])
        row[f"signal_Linf_{name}"] = float(signal_linf[index])
        row[f"relative_perturbation_L1_{name}"] = (
            float(perturbation_l1[index] / signal_l1[index])
            if signal_l1[index] > 100.0 * np.finfo(float).eps
            else float("nan")
        )
        row[f"relative_perturbation_Linf_{name}"] = (
            float(perturbation_linf[index] / signal_linf[index])
            if signal_linf[index] > 100.0 * np.finfo(float).eps
            else float("nan")
        )


def add_refinement_rates(
    rows: list[dict[str, object]], names: list[str]
) -> None:
    for row in rows:
        for name in names:
            row[f"rate_full_L1_{name}"] = float("nan")
            row[f"rate_perturbation_L1_{name}"] = float("nan")

    groups: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["amplitude"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(key=lambda item: int(item["Nx"]))
        for previous, current in zip(group[:-1], group[1:]):
            refinement = float(current["Nx"]) / float(previous["Nx"])
            for name in names:
                for prefix in ("full_L1", "perturbation_L1"):
                    previous_error = float(previous[f"{prefix}_{name}"])
                    current_error = float(current[f"{prefix}_{name}"])
                    if previous_error > 0.0 and current_error > 0.0:
                        current[f"rate_{prefix}_{name}"] = math.log(
                            previous_error / current_error
                        ) / math.log(refinement)


def add_background_rates(
    rows: list[dict[str, object]], names: list[str]
) -> None:
    previous: dict[str, object] | None = None
    for row in rows:
        for name in names:
            row[f"rate_L1_{name}"] = float("nan")
        if previous is not None:
            refinement = float(row["Nx"]) / float(previous["Nx"])
            for name in names:
                previous_error = float(previous[f"L1_{name}"])
                current_error = float(row[f"L1_{name}"])
                if previous_error > 0.0 and current_error > 0.0:
                    row[f"rate_L1_{name}"] = math.log(
                        previous_error / current_error
                    ) / math.log(refinement)
        previous = row


def save_snapshot(
    path: Path,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    continuous_equilibrium: np.ndarray,
    stored_equilibrium: np.ndarray,
    reference: np.ndarray,
    final_states: dict[str, np.ndarray],
) -> None:
    names = primitive_names(model.p.moments)
    continuous_primitive = primitive_array(
        continuous_equilibrium, model.p.moments
    )
    columns = [mesh.centers, np.asarray(model.bottom.value(mesh.centers))]
    headers = ["x", "bottom"]
    states = {
        "continuous_equilibrium": continuous_equilibrium,
        "stored_equilibrium": stored_equilibrium,
        "reference": reference,
        **final_states,
    }
    for prefix, state in states.items():
        primitive = primitive_array(state, model.p.moments)
        for column, name in enumerate(names):
            columns.extend(
                [
                    primitive[:, column],
                    primitive[:, column] - continuous_primitive[:, column],
                ]
            )
            headers.extend(
                [f"{name}_{prefix}", f"{name}_{prefix}_perturbation"]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", type=int, default=[100, 200, 400, 800])
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.05, 0.005])
    parser.add_argument("--center", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--reference", type=int, default=6400)
    parser.add_argument("--reference-check", type=int, default=3200)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_LABELS),
        default=list(METHOD_LABELS),
    )
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--source-cfl", type=float, default=0.5)
    parser.add_argument("--quadrature-points", type=int, default=4)
    parser.add_argument("--snapshot-grid", type=int, default=None)
    parser.add_argument("--skip-cfl-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=CASE_ROOT / "results")
    arguments = parser.parse_args()

    grids = sorted(set(arguments.grids))
    amplitudes = sorted(set(arguments.amplitudes), reverse=True)
    snapshot_grid = arguments.snapshot_grid or max(grids)
    if min(grids) < 2 or snapshot_grid not in grids:
        raise SystemExit("The grids and snapshot grid are inconsistent.")
    if arguments.reference <= arguments.reference_check:
        raise SystemExit("The primary reference must be finer than its check grid.")
    divisors = grids + [arguments.reference_check]
    if any(arguments.reference % cells != 0 for cells in divisors):
        raise SystemExit("Every grid must divide the primary reference grid.")
    if arguments.final_time <= 0.0 or arguments.cfl <= 0.0:
        raise SystemExit("Final time and CFL must be positive.")
    if arguments.sigma <= 0.0 or any(value <= 0.0 for value in amplitudes):
        raise SystemExit("Perturbation amplitudes and width must be positive.")
    if arguments.quadrature_points < 2:
        raise SystemExit("At least two cell quadrature points are required.")
    if not arguments.skip_cfl_check and "wb2" not in arguments.methods:
        raise SystemExit("The CFL check requires wb2 or --skip-cfl-check.")

    model, left_state = configured_problem()
    names = primitive_names(model.p.moments)
    fine_mesh = Mesh(arguments.reference)
    tight_stationary = stationary_solution(model, left_state, 1.0e-12, 1.0e-14)
    loose_stationary = stationary_solution(model, left_state, 1.0e-10, 1.0e-12)
    fine_continuous_equilibrium, _ = cell_average_initial_data(
        tight_stationary,
        fine_mesh,
        amplitude=0.0,
        center=arguments.center,
        sigma=arguments.sigma,
        quadrature_points=arguments.quadrature_points,
    )
    fine_loose_equilibrium, _ = cell_average_initial_data(
        loose_stationary,
        fine_mesh,
        amplitude=0.0,
        center=arguments.center,
        sigma=arguments.sigma,
        quadrature_points=arguments.quadrature_points,
    )
    stationary_tolerance_linf = float(
        np.max(np.abs(fine_loose_equilibrium - fine_continuous_equilibrium))
    )

    all_grids = sorted(set(grids + [arguments.reference_check, arguments.reference]))
    equilibria = {
        cells: build_equilibrium(model, Mesh(cells), left_state)
        for cells in all_grids
    }
    background_rows: list[dict[str, object]] = []
    for cells in grids:
        mesh = Mesh(cells)
        continuous_background = block_average(fine_continuous_equilibrium, cells)
        stored = equilibria[cells]
        l1_error, linf_error = component_errors(
            stored.cells, continuous_background, mesh.dx, model.p.moments
        )
        row: dict[str, object] = {
            "Nx": cells,
            "newton_residual_linf": float(np.max(stored.newton_residuals)),
            "newton_iterations_max": int(np.max(stored.newton_iterations)),
            "stored_equilibrium_sha256": state_checksum(stored.cells),
            "continuous_equilibrium_sha256": state_checksum(continuous_background),
        }
        for index, name in enumerate(names):
            row[f"L1_{name}"] = float(l1_error[index])
            row[f"Linf_{name}"] = float(linf_error[index])
        background_rows.append(row)
    add_background_rates(background_rows, names)

    accuracy_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    cfl_rows: list[dict[str, object]] = []
    reference_checksums: dict[str, dict[str, str]] = {}
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    for amplitude in amplitudes:
        tag = amplitude_tag(amplitude)
        print(
            f"Example 4: building common continuous initial data, amplitude={amplitude:g}",
            flush=True,
        )
        _, fine_initial = cell_average_initial_data(
            tight_stationary,
            fine_mesh,
            amplitude=amplitude,
            center=arguments.center,
            sigma=arguments.sigma,
            quadrature_points=arguments.quadrature_points,
        )

        print(f"Example 4: advancing wb2 reference Nx={arguments.reference}", flush=True)
        reference_snapshots, reference_steps = evolve_to_times(
            "wb2",
            model,
            fine_mesh,
            equilibria[arguments.reference],
            fine_initial,
            [arguments.final_time],
            cfl=arguments.cfl,
            source_cfl=arguments.source_cfl,
        )
        fine_reference = reference_snapshots[arguments.final_time]

        check_cells = arguments.reference_check
        check_mesh = Mesh(check_cells)
        check_initial = block_average(fine_initial, check_cells)
        print(f"Example 4: advancing reference check Nx={check_cells}", flush=True)
        check_snapshots, check_steps = evolve_to_times(
            "wb2",
            model,
            check_mesh,
            equilibria[check_cells],
            check_initial,
            [arguments.final_time],
            cfl=arguments.cfl,
            source_cfl=arguments.source_cfl,
        )
        restricted_reference = block_average(fine_reference, check_cells)
        restricted_background = block_average(
            fine_continuous_equilibrium, check_cells
        )
        reference_row: dict[str, object] = {
            "amplitude": amplitude,
            "coarse_reference_Nx": check_cells,
            "fine_reference_Nx": arguments.reference,
            "coarse_reference_steps": check_steps[arguments.final_time],
            "fine_reference_steps": reference_steps[arguments.final_time],
            "common_initial_sha256": state_checksum(check_initial),
        }
        add_error_columns(
            reference_row,
            check_snapshots[arguments.final_time],
            restricted_reference,
            restricted_background,
            check_mesh.dx,
            names,
            model.p.moments,
        )
        reference_rows.append(reference_row)
        reference_checksums[tag] = {
            "continuous_equilibrium": state_checksum(fine_continuous_equilibrium),
            "initial": state_checksum(fine_initial),
            "final": state_checksum(fine_reference),
        }

        snapshot_payload = None
        default_wb2_snapshot: np.ndarray | None = None
        for cells in grids:
            mesh = Mesh(cells)
            initial = block_average(fine_initial, cells)
            continuous_background = block_average(
                fine_continuous_equilibrium, cells
            )
            reference = block_average(fine_reference, cells)
            initial_checksum = state_checksum(initial)
            final_states: dict[str, np.ndarray] = {}

            for method in arguments.methods:
                print(
                    f"Example 4: advancing {METHOD_LABELS[method]}, "
                    f"Nx={cells}, amplitude={amplitude:g}",
                    flush=True,
                )
                snapshots, step_counts = evolve_to_times(
                    method,
                    model,
                    mesh,
                    equilibria[cells],
                    initial,
                    [arguments.final_time],
                    cfl=arguments.cfl,
                    source_cfl=arguments.source_cfl,
                )
                state = snapshots[arguments.final_time]
                row: dict[str, object] = {
                    "amplitude": amplitude,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "Nx": cells,
                    "final_time": arguments.final_time,
                    "steps": step_counts[arguments.final_time],
                    "minimum_height": float(np.min(state[:, 0])),
                    "initial_sha256": initial_checksum,
                    "continuous_equilibrium_sha256": state_checksum(
                        continuous_background
                    ),
                    "stored_equilibrium_sha256": state_checksum(
                        equilibria[cells].cells
                    ),
                }
                add_error_columns(
                    row,
                    state,
                    reference,
                    continuous_background,
                    mesh.dx,
                    names,
                    model.p.moments,
                )
                accuracy_rows.append(row)
                final_states[method] = state
                if cells == snapshot_grid and method == "wb2":
                    default_wb2_snapshot = state

            if cells == snapshot_grid:
                snapshot_payload = (
                    mesh,
                    continuous_background,
                    equilibria[cells].cells,
                    reference,
                    final_states,
                )

        if snapshot_payload is None:
            raise RuntimeError("The snapshot grid was not processed.")
        save_snapshot(
            arguments.output_dir
            / f"ex4_snapshot_a{tag}_nx{snapshot_grid}.csv",
            model,
            *snapshot_payload,
        )

        if not arguments.skip_cfl_check:
            if default_wb2_snapshot is None:
                raise RuntimeError("The default wb2 snapshot is unavailable.")
            mesh = Mesh(snapshot_grid)
            initial = block_average(fine_initial, snapshot_grid)
            half_snapshots, half_steps = evolve_to_times(
                "wb2",
                model,
                mesh,
                equilibria[snapshot_grid],
                initial,
                [arguments.final_time],
                cfl=0.5 * arguments.cfl,
                source_cfl=arguments.source_cfl,
            )
            cfl_l1, cfl_linf = component_errors(
                default_wb2_snapshot,
                half_snapshots[arguments.final_time],
                mesh.dx,
                model.p.moments,
            )
            cfl_row: dict[str, object] = {
                "amplitude": amplitude,
                "Nx": snapshot_grid,
                "CFL": arguments.cfl,
                "half_CFL": 0.5 * arguments.cfl,
                "half_CFL_steps": half_steps[arguments.final_time],
                "common_initial_sha256": state_checksum(initial),
            }
            for index, name in enumerate(names):
                cfl_row[f"L1_{name}"] = float(cfl_l1[index])
                cfl_row[f"Linf_{name}"] = float(cfl_linf[index])
            cfl_rows.append(cfl_row)

    add_refinement_rates(accuracy_rows, names)
    write_rows(arguments.output_dir / "ex4_accuracy.csv", accuracy_rows)
    write_rows(arguments.output_dir / "ex4_background.csv", background_rows)
    write_rows(arguments.output_dir / "ex4_reference_check.csv", reference_rows)
    if cfl_rows:
        write_rows(arguments.output_dir / "ex4_cfl_check.csv", cfl_rows)

    source_paths = [
        Path(__file__).with_name("quasi2d_wb.py"),
        Path(__file__),
        CASE_ROOT / "requirements.txt",
    ]
    manifest = {
        "example": "4",
        "date": "2026-09-04",
        "status": "completed by this invocation",
        "folder": "numerical-tests/swlme-quasi2d/perturbation-accuracy",
        "purpose": "accuracy for perturbations of the Example 3B equilibrium",
        "scope": "pseudo-two-dimensional: one x-dependent row with the full state",
        "parameters": model.p.metadata(),
        "bottom": {
            "amplitude": model.bottom.amplitude,
            "center": model.bottom.center,
            "width": model.bottom.width,
        },
        "left_primitive_state": [1.0, 0.4, 0.15, 0.08, -0.04, -0.03, 0.05],
        "source": (
            "standard projected Navier-slip source with gamma/epsilon and "
            "inverse_reynolds/(epsilon*h); no effective coefficient"
        ),
        "continuous_initial_value_problem": (
            "DOP853 stationary branch plus a Gaussian height perturbation; "
            "primitive velocities and moments retained; conservative cell "
            "averages formed on the primary grid and block-averaged to all grids"
        ),
        "stationary_reference_tolerances": {
            "tight_rtol": 1.0e-12,
            "tight_atol": 1.0e-14,
            "loose_rtol": 1.0e-10,
            "loose_atol": 1.0e-12,
            "fine_cell_average_linf_difference": stationary_tolerance_linf,
        },
        "grids": grids,
        "final_time": arguments.final_time,
        "amplitudes": amplitudes,
        "perturbation_center": arguments.center,
        "perturbation_sigma": arguments.sigma,
        "cell_quadrature_points": arguments.quadrature_points,
        "reference_Nx": arguments.reference,
        "reference_check_Nx": arguments.reference_check,
        "reference_checksums": reference_checksums,
        "methods": arguments.methods,
        "method_labels": METHOD_LABELS,
        "comparison_scope": (
            "standard versus wb1 is matched order; wb2 combines second order "
            "and well balancing and is not a balance-only comparison"
        ),
        "CFL": arguments.cfl,
        "source_CFL": arguments.source_cfl,
        "snapshot_grid": snapshot_grid,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": {
            str(path.relative_to(CASE_ROOT)): file_checksum(path)
            for path in source_paths
            if path.exists()
        },
    }
    (arguments.output_dir / "ex4_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved Example 4 results in {arguments.output_dir}", flush=True)


if __name__ == "__main__":
    main()
