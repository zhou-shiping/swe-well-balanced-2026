"""Run the complete lake-at-rest mesh and method comparison."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from lake_at_rest import (
    METHOD_LABELS,
    Mesh,
    ModelParameters,
    Quasi2DGSWLME,
    SmoothBottom,
    build_lake_equilibrium,
    diagnostic_values,
    evolve_to_times,
    primitive_array,
    residual_values,
)


TEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = TEST_ROOT / "results"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows were supplied for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_checksums() -> dict[str, str]:
    checksums = {}
    for path in sorted((TEST_ROOT / "code").glob("*.py")):
        checksums[f"code/{path.name}"] = sha256(path.read_bytes()).hexdigest()
    return checksums


def profile_rows(
    model: Quasi2DGSWLME,
    mesh: Mesh,
    equilibrium,
    final_states: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    equilibrium_primitive = primitive_array(model, equilibrium.cells)
    all_primitives = {
        method: primitive_array(model, state)
        for method, state in final_states.items()
    }
    rows = []
    for i, x in enumerate(mesh.centers):
        row: dict[str, object] = {
            "x": x,
            "bottom": equilibrium.bottom_cells[i],
            "free_surface_equilibrium": model.p.free_surface,
            "height_equilibrium": equilibrium_primitive[i, 0],
        }
        for method, primitive in all_primitives.items():
            row[f"height_{method}"] = primitive[i, 0]
            row[f"free_surface_{method}"] = (
                primitive[i, 0] + equilibrium.bottom_cells[i]
            )
            row[f"mean_u_{method}"] = primitive[i, 1]
            row[f"mean_v_{method}"] = primitive[i, 2]
            row[f"moments_max_{method}"] = np.max(np.abs(primitive[i, 3:]))
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> None:
    parameters = ModelParameters(moments=args.moments)
    model = Quasi2DGSWLME(parameters)
    bottom = SmoothBottom()
    methods = tuple(METHOD_LABELS)
    history_times = np.linspace(0.0, args.final_time, args.history_points)

    summary_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    history_final_states: dict[str, np.ndarray] = {}
    history_mesh = None
    history_equilibrium = None

    for cells in args.cells:
        mesh = Mesh(cells=cells)
        equilibrium = build_lake_equilibrium(model, mesh, bottom)
        output_times = history_times if cells == args.history_cells else [args.final_time]
        for method in methods:
            residual = residual_values(method, model, mesh, equilibrium)
            start = perf_counter()
            snapshots, step_counts = evolve_to_times(
                method,
                model,
                mesh,
                equilibrium,
                equilibrium.cells,
                output_times,
                cfl=args.cfl,
            )
            elapsed = perf_counter() - start
            final_state = snapshots[float(args.final_time)]
            final_diagnostics = diagnostic_values(
                model, mesh, equilibrium, final_state
            )
            summary_rows.append(
                {
                    "cells": cells,
                    "dx": mesh.dx,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "final_time": args.final_time,
                    "cfl": args.cfl,
                    "steps": step_counts[float(args.final_time)],
                    "runtime_seconds": elapsed,
                    **residual,
                    **final_diagnostics,
                }
            )

            if cells == args.history_cells:
                history_final_states[method] = final_state
                history_mesh = mesh
                history_equilibrium = equilibrium
                for time in history_times:
                    diagnostics = diagnostic_values(
                        model, mesh, equilibrium, snapshots[float(time)]
                    )
                    history_rows.append(
                        {
                            "cells": cells,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "time": time,
                            "steps": step_counts[float(time)],
                            **diagnostics,
                        }
                    )

    if history_mesh is None or history_equilibrium is None:
        raise ValueError("The history mesh must be included in the mesh sequence.")

    results_directory = args.results_dir.resolve()
    write_csv(results_directory / "lake_rest_summary.csv", summary_rows)
    write_csv(results_directory / "lake_rest_history.csv", history_rows)
    write_csv(
        results_directory / f"lake_rest_profile_nx{args.history_cells}.csv",
        profile_rows(
            model,
            history_mesh,
            history_equilibrium,
            history_final_states,
        ),
    )

    manifest = {
        "test": "quasi-2D G-SWLME lake at rest",
        "created_or_updated": "2026-09-04",
        "repository_relative_path": "numerical-tests/swlme-quasi2d/lake-at-rest/",
        "status": "completed; verification is recorded separately",
        "model": {
            "state_size": model.size,
            "moments": parameters.moments,
            "gravity": parameters.gravity,
            "free_surface": parameters.free_surface,
            "source_convention": "frictionless; bed slope only",
            "y_dependence": "none; all vector components retained",
        },
        "bottom": {
            "formula": (
                "0.15 + 0.20 exp(-100 (x-0.32)^2) + "
                "0.10 exp(-200 (x-0.70)^2) + 0.03 sin(2 pi x)"
            ),
            "discrete_representation": (
                "continuous piecewise linear through analytic face samples"
            ),
        },
        "run": {
            "domain": [0.0, 1.0],
            "cells": args.cells,
            "history_cells": args.history_cells,
            "history_points": args.history_points,
            "final_time": args.final_time,
            "cfl": args.cfl,
            "path_quadrature_points": 4,
            "methods": METHOD_LABELS,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_sha256": source_checksums(),
        "artifacts": [
            "results/lake_rest_summary.csv",
            "results/lake_rest_history.csv",
            f"results/lake_rest_profile_nx{args.history_cells}.csv",
        ],
    }
    manifest_path = results_directory / "lake_rest_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("cells method residual_inf eta_drift_inf velocity_inf moments_inf min_depth")
    for row in summary_rows:
        velocity = max(float(row["mean_u_linf"]), float(row["mean_v_linf"]))
        print(
            f"{int(row['cells']):5d} {str(row['method']):8s} "
            f"{float(row['residual_full_linf']):.6e} "
            f"{float(row['free_surface_linf']):.6e} "
            f"{velocity:.6e} "
            f"{float(row['moments_linf']):.6e} "
            f"{float(row['minimum_depth']):.6e}"
        )
    print(f"Wrote {results_directory}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=[50, 100, 200, 400])
    parser.add_argument("--history-cells", type=int, default=200)
    parser.add_argument("--history-points", type=int, default=41)
    parser.add_argument("--moments", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=10.0)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
