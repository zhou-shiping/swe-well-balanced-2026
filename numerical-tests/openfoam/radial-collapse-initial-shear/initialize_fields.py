"""Stream analytic cell-center initial fields for the single Cartesian block.

Run after blockMesh and before decomposePar. No full 3D array is allocated.
The ordering is OpenFOAM blockMesh x-fastest, y-next, z-slowest, as in the
baseline exporter. This utility refuses to replace already initialized data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from problem import (DOMAIN_M, GRAVITY_M_S2, RHO_AIR, RHO_WATER,
                     case_settings, depth, initial_velocity)


def load_settings(case):
    settings = json.loads((case/"case_settings.json").read_text())
    nx, ny, nz = settings["mesh_cells_openfoam_xyz"]
    expected = case_settings(ny, nx, nz)
    source_hash = hashlib.sha256(Path(__file__).with_name("problem.py").read_bytes()).hexdigest()
    if source_hash != settings["problem_sha256"]:
        raise ValueError("problem.py changed after case preparation")
    for key in ("domain_m", "center_m", "dam_radius_m", "inner_depth_m",
                "outer_depth_m", "shear_radius_m", "peak_mean_speed_m_s",
                "cosine_amplitudes", "gravity_m_s2", "rho_water", "rho_air"):
        if settings[key] != expected[key]:
            raise ValueError(f"case/source settings mismatch: {key}; prepare a new case")
    text = (case/"system/blockMeshDict").read_text()
    counts = re.findall(r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", text)
    if len(counts) != 1 or tuple(map(int, counts[0])) != (nx, ny, nz):
        raise ValueError("blockMeshDict and case_settings.json disagree")
    # Reject geometry edits as well as cell-count edits: the stream order and
    # analytic coordinates rely on exactly the generated uniform block.
    mesh_hash = hashlib.sha256((case/"system/blockMeshDict").read_bytes()).hexdigest()
    if mesh_hash != settings["input_sha256"]["system/blockMeshDict"]:
        raise ValueError("blockMeshDict changed after preparation")
    for relative, original_hash in settings["input_sha256"].items():
        if relative.startswith(("system/", "constant/")):
            actual = hashlib.sha256((case/relative).read_bytes()).hexdigest()
            if actual != original_hash:
                raise ValueError(f"case dictionary changed after preparation: {relative}")
    return settings


def initialize(case):
    settings = load_settings(case)
    if (case/"initialization.json").exists() or (case/"processor0").exists():
        raise FileExistsError("refusing to reinitialize an initialized/decomposed case")
    nx, ny, nz = settings["mesh_cells_openfoam_xyz"]
    x = (np.arange(nx)+0.5)*DOMAIN_M[0]/nx
    y = (np.arange(ny)+0.5)*DOMAIN_M[1]/ny
    z = (np.arange(nz)+0.5)*DOMAIN_M[2]/nz
    handles, tails, staged = {}, {}, {}
    # Check all placeholders before opening any output.
    texts = {name: (case/"0"/name).read_text() for name in ("U", "alpha.water", "p_rgh")}
    for name, text in texts.items():
        if "internalField uniform" not in text:
            raise FileExistsError(f"{name} is not an untouched input placeholder")
    try:
        for name, text in texts.items():
            prefix, rest = text.split("internalField", 1)
            tails[name] = rest.split(";", 1)[1]
            staged[name] = case/"0"/f".{name}.initializing"
            handles[name] = staged[name].open("x")
            kind = "vector" if name == "U" else "scalar"
            handles[name].write(prefix+f"internalField nonuniform List<{kind}>\n{nx*ny*nz}\n(\n")
        volume = 0.0
        for zi in z:
            h = depth(x, zi)
            alpha = np.broadcast_to(y[:, None] < h[None, :], (ny, nx)).astype(float)
            velocity = initial_velocity(x[None, :], y[:, None], zi)
            # p=0 at the atmospheric top; gh=-g*y. This is an initial
            # hydrostatic guess, not a centrifugal-equilibrium pressure.
            prgh = RHO_AIR*GRAVITY_M_S2*DOMAIN_M[1] + (
                RHO_WATER-RHO_AIR)*GRAVITY_M_S2*h[None, :]*alpha
            np.savetxt(handles["U"], velocity.reshape(-1, 3), fmt="(%.12g %.12g %.12g)")
            np.savetxt(handles["alpha.water"], alpha.ravel(), fmt="%.12g")
            np.savetxt(handles["p_rgh"], prgh.ravel(), fmt="%.12g")
            volume += float(alpha.sum())*np.prod(DOMAIN_M)/(nx*ny*nz)
        for name, handle in handles.items():
            handle.write(");\n"+tails[name])
            handle.close()
        for name, path in staged.items():
            path.replace(case/"0"/name)
    finally:
        for handle in handles.values():
            handle.close()
    report = {"status": "initial fields written; not a solver result",
              "cell_count": nx*ny*nz, "water_volume_m3": volume,
              "coordinate_order": "OpenFOAM x,y,z; physical vertical is y",
              "sampling": "analytic cell-center samples; x fastest",
              "problem_sha256": hashlib.sha256(Path(__file__).with_name("problem.py").read_bytes()).hexdigest()}
    (case/"initialization.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    initialize(parser.parse_args().case.resolve())
