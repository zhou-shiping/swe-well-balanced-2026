"""Run Example 3B: full pseudo-2D moving equilibrium with standard friction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


CASE_ROOT = Path(__file__).resolve().parents[1]
SHARED_CODE = CASE_ROOT.parent / "code"
sys.path.insert(0, str(SHARED_CODE))

from quasi2d_wb import (  # noqa: E402
    METHOD_LABELS,
    GaussianBottom,
    Mesh,
    Parameters,
    Quasi2DGSWLME,
    branch_margins,
    build_equilibrium,
    component_errors,
    conservative_state,
    evolve_to_times,
    primitive_array,
    primitive_names,
    spatial_operator,
    state_checksum,
)


def configured_problem() -> tuple[Quasi2DGSWLME, np.ndarray]:
    parameters = Parameters(
        moments=2,
        gravity=1.0,
        epsilon=0.1,
        gamma=0.002,
        inverse_reynolds=0.0005,
    )
    model = Quasi2DGSWLME(parameters, GaussianBottom())
    left_state = conservative_state(
        height=1.0,
        mean_u=0.4,
        mean_v=0.15,
        alpha=[0.08, -0.03],
        beta=[-0.04, 0.05],
    )
    return model, left_state


def adaptive_reference(
    model: Quasi2DGSWLME,
    left_state: np.ndarray,
    points: np.ndarray,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Integrate the stationary ODE independently with adaptive DOP853."""
    solution = solve_ivp(
        lambda x, state: model.stationary_slope(state, float(x)),
        (float(points.min()) if points.min() < 0.0 else 0.0, 1.0),
        left_state,
        method="DOP853",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"The adaptive stationary reference failed: {solution.message}")
    reference = solution.sol(points).T
    if np.min(reference[:, 0]) <= 0.0 or not np.isfinite(reference).all():
        raise RuntimeError("The adaptive stationary reference is inadmissible.")
    return reference, {
        "nfev": int(solution.nfev),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def add_refinement_rates(
    rows: list[dict[str, object]], component_names: list[str]
) -> None:
    previous: dict[str, float] | None = None
    previous_cells: int | None = None
    for row in rows:
        cells = int(row["Nx"])
        for name in component_names:
            error = float(row[f"L1_{name}"])
            rate_name = f"rate_L1_{name}"
            if previous is None or previous_cells is None or error == 0.0:
                row[rate_name] = ""
            else:
                row[rate_name] = np.log(previous[name] / error) / np.log(
                    cells / previous_cells
                )
        previous = {
            name: float(row[f"L1_{name}"]) for name in component_names
        }
        previous_cells = cells


def save_snapshot(
    path: Path,
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium: np.ndarray,
    reference: np.ndarray,
    final_states: dict[str, np.ndarray],
) -> None:
    names = primitive_names(model.p.moments)
    columns = [
        mesh.centers,
        np.asarray(model.bottom.value(mesh.centers)),
        primitive_array(equilibrium, model.p.moments)[:, 0]
        + model.bottom.value(mesh.centers),
    ]
    headers = ["x", "bottom", "surface_equilibrium"]
    for prefix, state in (("equilibrium", equilibrium), ("reference", reference)):
        primitive = primitive_array(state, model.p.moments)
        for column, name in enumerate(names):
            columns.append(primitive[:, column])
            headers.append(f"{name}_{prefix}")
    for method, state in final_states.items():
        primitive = primitive_array(state, model.p.moments)
        for column, name in enumerate(names):
            columns.append(primitive[:, column])
            headers.append(f"{name}_{method}")
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
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=[0.0] + [float(value) for value in range(1, 11)],
    )
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_LABELS), default=list(METHOD_LABELS)
    )
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--source-cfl", type=float, default=0.5)
    parser.add_argument("--snapshot-grid", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=CASE_ROOT / "results")
    arguments = parser.parse_args()

    grids = sorted(set(arguments.grids))
    times = sorted(set(arguments.times))
    snapshot_grid = arguments.snapshot_grid or max(grids)
    if snapshot_grid not in grids:
        raise SystemExit("The snapshot grid must be present in --grids.")
    if min(grids) < 2 or min(times) < 0.0:
        raise SystemExit("Grids and output times must be valid.")

    model, left_state = configured_problem()
    names = primitive_names(model.p.moments)
    equilibrium_rows: list[dict[str, object]] = []
    preservation_rows: list[dict[str, object]] = []
    snapshot_payload: tuple[Mesh, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None = None

    for cells in grids:
        print(f"Example 3B: constructing Nx={cells}", flush=True)
        mesh = Mesh(cells)
        equilibrium = build_equilibrium(model, mesh, left_state)
        reference, reference_data = adaptive_reference(
            model, left_state, mesh.centers, 1.0e-12, 1.0e-14
        )
        loose_reference, loose_data = adaptive_reference(
            model, left_state, mesh.centers, 1.0e-10, 1.0e-12
        )
        l1_equilibrium, linf_equilibrium = component_errors(
            equilibrium.cells, reference, mesh.dx, model.p.moments
        )
        reference_l1, reference_linf = component_errors(
            loose_reference, reference, mesh.dx, model.p.moments
        )
        margins = branch_margins(model, equilibrium.cells)
        equilibrium_row: dict[str, object] = {
            "Nx": cells,
            "representation": "midpoint collocation values",
            "newton_residual_linf": float(np.max(equilibrium.newton_residuals)),
            "newton_iterations_max": int(np.max(equilibrium.newton_iterations)),
            "reference_nfev_tight": reference_data["nfev"],
            "reference_nfev_loose": loose_data["nfev"],
            "reference_tolerance_L1_max": float(np.max(reference_l1)),
            "reference_tolerance_Linf_max": float(np.max(reference_linf)),
            **margins,
            "equilibrium_sha256": state_checksum(equilibrium.cells),
        }
        for index, name in enumerate(names):
            equilibrium_row[f"L1_{name}"] = float(l1_equilibrium[index])
            equilibrium_row[f"Linf_{name}"] = float(linf_equilibrium[index])
        equilibrium_rows.append(equilibrium_row)

        residuals = {
            method: float(
                np.max(
                    np.abs(
                        spatial_operator(
                            method, model, mesh, equilibrium, equilibrium.cells
                        )
                    )
                )
            )
            for method in arguments.methods
        }
        final_states: dict[str, np.ndarray] = {}
        for method in arguments.methods:
            print(
                f"Example 3B: advancing {METHOD_LABELS[method]}, Nx={cells}",
                flush=True,
            )
            snapshots, step_counts = evolve_to_times(
                method,
                model,
                mesh,
                equilibrium,
                equilibrium.cells,
                times,
                cfl=arguments.cfl,
                source_cfl=arguments.source_cfl,
            )
            for time in times:
                state = snapshots[time]
                l1_drift, linf_drift = component_errors(
                    state, equilibrium.cells, mesh.dx, model.p.moments
                )
                row: dict[str, object] = {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "Nx": cells,
                    "time": time,
                    "steps": step_counts[time],
                    "initial_spatial_residual_linf": residuals[method],
                    "maximum_component_L1_drift": float(np.max(l1_drift)),
                    "minimum_height": float(np.min(state[:, 0])),
                    "equilibrium_sha256": state_checksum(equilibrium.cells),
                }
                for index, name in enumerate(names):
                    row[f"L1_{name}"] = float(l1_drift[index])
                    row[f"Linf_{name}"] = float(linf_drift[index])
                preservation_rows.append(row)
            final_states[method] = snapshots[times[-1]]

        if cells == snapshot_grid:
            snapshot_payload = (
                mesh,
                equilibrium.cells,
                reference,
                final_states,
            )

    add_refinement_rates(equilibrium_rows, names)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(arguments.output_dir / "ex3b_equilibrium.csv", equilibrium_rows)
    write_rows(arguments.output_dir / "ex3b_preservation.csv", preservation_rows)
    if snapshot_payload is None:
        raise RuntimeError("The requested snapshot grid was not processed.")
    save_snapshot(
        arguments.output_dir / f"ex3b_snapshot_nx{snapshot_grid}.csv",
        model,
        snapshot_payload[0],
        snapshot_payload[1],
        snapshot_payload[2],
        snapshot_payload[3],
    )

    source_paths = [
        SHARED_CODE / "quasi2d_wb.py",
        Path(__file__),
    ]
    manifest = {
        "example": "3B",
        "status": "completed by this invocation",
        "date": "2026-09-04",
        "folder": "numerical-tests/swlme-quasi2d/stationary-preservation/example3b",
        "purpose": "full pseudo-2D equilibrium accuracy and stationary preservation",
        "representation": "midpoint collocation values used as approximate cell states",
        "parameters": model.p.metadata(),
        "bottom": {
            "amplitude": model.bottom.amplitude,
            "center": model.bottom.center,
            "width": model.bottom.width,
        },
        "left_primitive_state": [1.0, 0.4, 0.15, 0.08, -0.04, -0.03, 0.05],
        "grids": grids,
        "times": times,
        "methods": arguments.methods,
        "method_labels": METHOD_LABELS,
        "CFL": arguments.cfl,
        "source_CFL": arguments.source_cfl,
        "reference": "adaptive DOP853 stationary ODE with tolerance refinement",
        "source": (
            "standard projected Navier-slip source with gamma/epsilon and "
            "inverse_reynolds/(epsilon*h); no effective coefficient"
        ),
        "source_sha256": {
            str(path.relative_to(CASE_ROOT.parent)): file_checksum(path)
            for path in source_paths
            if path.exists()
        },
    }
    (arguments.output_dir / "ex3b_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved Example 3B results in {arguments.output_dir}", flush=True)


if __name__ == "__main__":
    main()
