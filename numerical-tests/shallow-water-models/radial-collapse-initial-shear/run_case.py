"""Run one independently evolved moment order; never overwrite an earlier run."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from moment_solver import stable_timestep, step, validate
from problem import (CENTER_M, GRAVITY_M_S2, LENGTH_M, OPENFOAM_PROBLEM_SHA256,
                     OUTPUT_TIMES_S, initial_state, moment_factors)

HERE = Path(__file__).resolve().parent


def diagnostics(state, x, y):
    h = state[..., 0]
    primitive = state[..., 1:]/h[..., None]
    area = LENGTH_M**2/(len(x)*len(y))
    weights = np.repeat(1/(2*np.arange(primitive.shape[-1]//2)+1), 2)
    energy = (h*np.sum(primitive**2*weights, axis=-1)+GRAVITY_M_S2*h*h)/2
    X, Y = x[:, None]-CENTER_M, y[None, :]-CENTER_M
    edge = np.hypot(X, Y) >= 45
    result = {
        "volume_m3": float(h.sum()*area),
        "momentum_x_m4_s": float(state[..., 1].sum()*area),
        "momentum_y_m4_s": float(state[..., 2].sum()*area),
        "angular_momentum_m5_s": float(np.sum(X*state[..., 2]-Y*state[..., 1])*area),
        "mechanical_energy_m5_s2": float(energy.sum()*area),
        "min_depth_m": float(h.min()), "max_depth_m": float(h.max()),
        "max_mean_speed_m_s": float(np.linalg.norm(primitive[..., :2], axis=-1).max()),
        "far_field_depth_deviation_m": float(np.max(abs(h[edge]-1))),
        "far_field_speed_m_s": float(np.linalg.norm(primitive[..., :2][edge], axis=-1).max()),
    }
    for j in range((state.shape[-1]-3)//2):
        result[f"max_abs_alpha_{j+1}_m_s"] = float(abs(primitive[..., 2+2*j]).max())
        result[f"max_abs_beta_{j+1}_m_s"] = float(abs(primitive[..., 3+2*j]).max())
    return result


def run(nx=400, ny=None, order=2, final_time=3.0, cfl=0.3,
        output_root=HERE/'results', quadrature=8, spatial_order=2):
    ny = nx if ny is None else ny
    if not np.isfinite(final_time) or final_time <= 0 or not 0 < cfl <= 0.5:
        raise ValueError("need finite positive final_time and 0<cfl<=0.5")
    if spatial_order not in (1, 2):
        raise ValueError("spatial_order must be 1 or 2")
    x, y, state = initial_state(nx, ny, order, quadrature)
    validate(state)
    label = f'N{order}_{nx}x{ny}_cfl{cfl:g}_T{final_time:g}_s{spatial_order}_q{quadrature}'
    directory = Path(output_root).resolve()/label
    directory.mkdir(parents=True, exist_ok=False)
    times = [t for t in OUTPUT_TIMES_S if t <= final_time]
    if times[-1] != final_time:
        times.append(final_time)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "equations": "frictionless flat-bottom 2D G-SWLME; both moment families evolved",
        "state_ordering": "h,hu_m,hv_m,h*alpha_1,h*beta_1,...",
        "array_axes": ["x", "y", "component"], "horizontal_coordinates": ["x_m", "y_m"],
        "domain_m": [0, LENGTH_M, 0, LENGTH_M], "gravity_m_s2": GRAVITY_M_S2,
        "order": order, "nx": nx, "ny": ny, "cfl": cfl, "final_time_s": final_time,
        "spatial_order": spatial_order, "initial_quadrature": quadrature,
        "initial_moment_factors": moment_factors(order).tolist(),
        "initial_vertical_projection": "analytic modes 1-3; max(64,N+8)-point Gauss for higher modes",
        "initial_averaging": "analytic circle area for h; tensor Gauss conservative momenta",
        "initial_openfoam_specification_sha256": OPENFOAM_PROBLEM_SHA256,
        "boundary": "impermeable free-slip; reverse normal mean and moment pairs",
        "method": "unsplit MUSCL generalized-minmod(theta=1.5), path-conservative LLF, SSP-RK2",
        "path": "straight conservative-state segments, three-point Gauss on faces and inside cells",
        "requested_times_s": times, "snapshots": [],
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))},
        "validation_scope": "completion is not convergence or OpenFOAM-reference qualification",
    }
    metadata_path = directory/'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    start = perf_counter()
    time, steps = 0.0, 0
    dx, dy = LENGTH_M/nx, LENGTH_M/ny
    initial_volume = diagnostics(state, x, y)['volume_m3']
    minimum_depth = float(state[..., 0].min())
    try:
        with (directory/'diagnostics.csv').open('w', newline='') as stream:
            writer = None
            for index, target in enumerate(times):
                while time < target:
                    dt = min(stable_timestep(state, dx, dy, cfl, spatial_order), target-time)
                    if not np.isfinite(dt) or dt <= 0 or time+dt == time:
                        raise FloatingPointError("time step is not positive/representable")
                    state = step(state, dt, dx, dy, spatial_order)
                    time = min(time+dt, target)
                    steps += 1
                    minimum_depth = min(minimum_depth, float(state[..., 0].min()))
                record = {"time_s": target, "steps": steps, **diagnostics(state, x, y)}
                record['relative_volume_defect'] = (record['volume_m3']-initial_volume)/initial_volume
                record['minimum_depth_over_run_m'] = minimum_depth
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                stream.flush()
                filename = f'snapshot_{index:03d}.npz'
                np.savez_compressed(directory/filename, x_m=x, y_m=y, time_s=target, state=state)
                metadata['snapshots'].append({"time_s": target, "file": filename})
                metadata['elapsed_seconds'] = perf_counter()-start
                metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
                print(f'{label}: t={target:g}, steps={steps}, h_min={record["min_depth_m"]:.8g}, '
                      f'volume defect={record["relative_volume_defect"]:.3e}', flush=True)
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error_type=type(error).__name__, error_message=str(error),
                        failure_time_s=time, steps=steps)
        raise
    finally:
        metadata['elapsed_seconds'] = perf_counter()-start
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, default=400)
    parser.add_argument('--ny', type=int)
    parser.add_argument('--order', type=int, required=True, help='nonnegative moment order N')
    parser.add_argument('--final-time', type=float, default=3.0)
    parser.add_argument('--cfl', type=float, default=0.3)
    parser.add_argument('--quadrature', type=int, default=8)
    parser.add_argument('--spatial-order', type=int, choices=(1, 2), default=2)
    parser.add_argument('--output-root', type=Path, default=HERE/'results')
    print(run(**vars(parser.parse_args())))
