#!/usr/bin/env python3
"""Run one reduced-model radial-collapse case and save compact evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from collapse_solver import (
    DOMAIN_LENGTH_M,
    GRAVITY_M_PER_S2,
    OUTPUT_TIMES_S,
    solve,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--order", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--final-time", type=float, default=3.0)
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def write_diagnostics(
    path: Path,
    times_s: np.ndarray,
    records: list[dict[str, float]],
) -> None:
    fieldnames = ["time_s", *records[0].keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for time_s, record in zip(times_s, records, strict=True):
            writer.writerow({"time_s": time_s, **record})


def main() -> None:
    arguments = parse_arguments()
    nz = arguments.nx if arguments.nz is None else arguments.nz
    if arguments.nx < 4 or nz < 4:
        raise SystemExit("nx and nz must both be at least four")
    if arguments.cfl <= 0.0 or arguments.final_time <= 0.0:
        raise SystemExit("cfl and final-time must be positive")

    output_times = tuple(
        time_s for time_s in OUTPUT_TIMES_S if time_s <= arguments.final_time
    )
    if output_times[-1] != arguments.final_time:
        output_times = output_times + (arguments.final_time,)
    output_label = (
        f"collapse_o{arguments.order}_nx{arguments.nx}_nz{nz}"
    )
    print(
        f"{output_label}: CFL={arguments.cfl}, "
        f"final_time={arguments.final_time} s",
        flush=True,
    )
    if arguments.preflight:
        print("PREPARED: arguments accepted; no simulation run", flush=True)
        return

    start = perf_counter()
    x, z, times, states, diagnostics = solve(
        arguments.nx,
        nz,
        order=arguments.order,
        final_time=arguments.final_time,
        cfl=arguments.cfl,
        output_times=output_times,
    )
    elapsed_seconds = perf_counter() - start
    results_directory = arguments.results_directory.resolve()
    results_directory.mkdir(parents=True, exist_ok=True)

    data_path = results_directory / f"{output_label}.npz"
    np.savez_compressed(
        data_path,
        x_m=x,
        z_m=z,
        times_s=times,
        state=states,
    )
    diagnostics_path = results_directory / f"{output_label}_diagnostics.csv"
    write_diagnostics(diagnostics_path, times, diagnostics)

    metadata = {
        "created": "2026-09-04",
        "equations": "frictionless flat-bottom two-dimensional G-SWLME",
        "specialization": (
            "zero initial moments; invariant SWE subsystem for every order"
        ),
        "state_ordering": "h, h*u, h*v, h*alpha_1, h*beta_1, ...",
        "horizontal_coordinates": ["x", "z"],
        "domain_m": [0.0, DOMAIN_LENGTH_M, 0.0, DOMAIN_LENGTH_M],
        "gravity_m_per_s2": GRAVITY_M_PER_S2,
        "initial_depth_m": {
            "inside_radius_15_m": 1.5,
            "outside_radius_15_m": 1.0,
        },
        "initial_velocity_m_per_s": 0.0,
        "initial_moments_m_per_s": 0.0,
        "boundary_condition": "impermeable free-slip on all four sides",
        "spatial_scheme": "MUSCL-MC with local Lax-Friedrichs flux",
        "time_integrator": "SSP-RK2",
        "cfl": arguments.cfl,
        "nx": arguments.nx,
        "nz": nz,
        "moment_order": arguments.order,
        "times_s": times.tolist(),
        "elapsed_seconds": elapsed_seconds,
        "numpy_version": np.__version__,
        "data_file": data_path.name,
        "diagnostics_file": diagnostics_path.name,
    }
    metadata_path = results_directory / f"{output_label}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved {data_path}", flush=True)
    print(f"saved {diagnostics_path}", flush=True)
    print(f"saved {metadata_path}", flush=True)
    print(f"elapsed_seconds={elapsed_seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
