"""Run one local G-SWLME case for comparison with the parent system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moment_solver import initial_state, solve


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
    _, _, initial_state_values = initial_state(arguments.nx, ny, arguments.order)
    x, y, state, diagnostics = solve(
        nx=arguments.nx,
        ny=ny,
        order=arguments.order,
        final_time=arguments.final_time,
        gravity=arguments.gravity,
        cfl=arguments.cfl,
    )
    stem = f"moment_N{arguments.order}_{arguments.nx}x{ny}_t{arguments.final_time:g}"
    data_path = arguments.output_directory / f"{stem}.npz"
    metadata_path = arguments.output_directory / f"{stem}.json"
    np.savez_compressed(
        data_path,
        x=x,
        y=y,
        state=state,
        initial_state=initial_state_values,
        order=arguments.order,
        nx=arguments.nx,
        ny=ny,
        final_time=arguments.final_time,
        gravity=arguments.gravity,
        cfl=arguments.cfl,
    )
    metadata = {
        "created_by": "example1b/run_moment.py",
        "equations": "frictionless flat-bottom two-dimensional G-SWLME",
        "boundary_condition": "periodic in x and y",
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
