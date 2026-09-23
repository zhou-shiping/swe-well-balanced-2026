"""Run one vertically resolved parent-system case and save projected moments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from parent_solver import initial_state, project_moments, solve


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    ny = arguments.ny or arguments.nx
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    _, _, _, initial_height, initial_momentum = initial_state(
        arguments.nx, ny, arguments.nz
    )
    initial_projected_state = project_moments(
        initial_height, initial_momentum, maximum_order=2
    )
    del initial_height, initial_momentum
    x, y, zeta, height, momentum, diagnostics = solve(
        nx=arguments.nx,
        ny=ny,
        nz=arguments.nz,
        final_time=arguments.final_time,
        gravity=arguments.gravity,
        cfl=arguments.cfl,
    )
    projected_state = project_moments(height, momentum, maximum_order=2)
    cfl_label = f"{arguments.cfl:.6g}".replace(".", "p")
    stem = (
        f"reference_{arguments.nx}x{ny}x{arguments.nz}"
        f"_cfl{cfl_label}_t{arguments.final_time:g}"
    )
    data_path = arguments.output_directory / f"{stem}.npz"
    metadata_path = arguments.output_directory / f"{stem}.json"
    np.savez_compressed(
        data_path,
        x=x,
        y=y,
        zeta=zeta,
        height=height,
        momentum=momentum,
        projected_state=projected_state,
        initial_projected_state=initial_projected_state,
        nx=arguments.nx,
        ny=ny,
        nz=arguments.nz,
        final_time=arguments.final_time,
        gravity=arguments.gravity,
        cfl=arguments.cfl,
    )
    metadata = {
        "created_by": "example1b/run_reference.py",
        "equations": "frictionless flat-bottom vertically resolved mapped hydrostatic system",
        "boundary_condition": (
            "periodic in x and y, zero mapped vertical mass flux at zeta endpoints"
        ),
        "spatial_scheme": (
            "MUSCL common-speed Rusanov in x-y and upwind vertical momentum transport"
        ),
        "vertical_coupling": "stagewise conservative continuity recurrence",
        "time_integrator": "SSP-RK3",
        "vertical_quadrature": "uniform midpoint rule",
        "initial_projection_saved": True,
        "nx": arguments.nx,
        "ny": ny,
        "nz": arguments.nz,
        "gravity": arguments.gravity,
        "cfl": arguments.cfl,
        **diagnostics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"saved {data_path}")


if __name__ == "__main__":
    main()
