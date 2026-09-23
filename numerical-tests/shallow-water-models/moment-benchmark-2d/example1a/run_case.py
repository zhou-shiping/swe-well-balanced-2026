"""Run one fixed-order G-SWLME discretization case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moment_solver import solve


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    ny = arguments.ny or arguments.nx
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    x, y, state, diagnostics = solve(
        nx=arguments.nx,
        ny=ny,
        order=arguments.order,
        final_time=arguments.final_time,
        gravity=arguments.gravity,
        cfl=arguments.cfl,
    )

    cfl_label = f"{arguments.cfl:.6g}".replace(".", "p")
    stem = (
        f"g_swlme_N{arguments.order}_{arguments.nx}x{ny}"
        f"_cfl{cfl_label}_t{arguments.final_time:g}"
    )
    data_path = arguments.output_directory / f"{stem}.npz"
    metadata_path = arguments.output_directory / f"{stem}.json"
    np.savez_compressed(
        data_path,
        x=x,
        y=y,
        state=state,
        order=arguments.order,
        nx=arguments.nx,
        ny=ny,
        final_time=arguments.final_time,
        gravity=arguments.gravity,
        cfl=arguments.cfl,
    )
    metadata = {
        "created_by": "example1a/run_case.py",
        "equations": "frictionless flat-bottom two-dimensional G-SWLME",
        "boundary_condition": "periodic in x and y",
        "spatial_scheme": (
            "MUSCL local Lax-Friedrichs plus centered smooth nonconservative products"
        ),
        "time_integrator": "SSP-RK3",
        "moment_order": arguments.order,
        "nx": arguments.nx,
        "ny": ny,
        "gravity": arguments.gravity,
        "cfl": arguments.cfl,
        **diagnostics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"saved {data_path}")


if __name__ == "__main__":
    main()
