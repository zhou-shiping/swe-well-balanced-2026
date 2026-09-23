# OpenFOAM: collapse from rest

Updated: 2026-09-23.
Repository-relative path: `numerical-tests/openfoam/radial-collapse-perfect-slip/`.

The original N200, N400, and N800 case layout is retained.
The OpenFOAM physical vertical direction is $y$; N200, N400, and N800 count vertical cells, independently of moment order in the Python models.
Each case starts at time zero and advances to $3\,\mathrm{s}$ with saved fields every $0.5\,\mathrm{s}$.
Only simulation inputs and data-generation code are included.

## Local prerequisites

Run inside a local OpenFOAM Foundation shell providing `foamRun` with the `incompressibleVoF` solver.
The input dictionaries identify Foundation version 12.
Use a configured OpenFOAM installation or its local Linux environment; the Python environment alone does not provide OpenFOAM.
Source your installation's environment before the commands below so `WM_PROJECT_DIR` and the OpenFOAM executables are available.
A parallel run also requires a compatible `mpirun`.
The [OpenFOAM Foundation parallel-run guide](https://doc.cfd.direct/openfoam/user-guide-v12/running-applications-parallel) explains decomposition and the MPI command used below.
No cluster module commands, scheduler, account, or submission scripts are required.

| Case folder | Mesh $(x,y,z)$ | Cells | MPI ranks in supplied dictionary |
|---|---|---:|---:|
| `water_collapse_parallel_N200/` | $400\times200\times400$ | 32,000,000 | 8 |
| `water_collapse_parallel_N400/` | $400\times400\times400$ | 64,000,000 | 16 |
| `water_collapse_parallel_N800/` | $400\times800\times400$ | 128,000,000 | 32 |

These are production meshes with tens of millions of cells, not laptop-sized trial cases.
Check local RAM, disk space, and available CPU ranks before starting a case.
The serial and parallel workflows below are alternatives for a fresh case.
Do not run both sequentially in the same case directory.

## Enter a case

From the repository root:

```bash
cd numerical-tests/openfoam/radial-collapse-perfect-slip/water_collapse_parallel_N200
```

Select the N400 or N800 folder instead for the other meshes.
Every remaining command is run inside the selected case folder.

## Serial data generation

```bash
./Allrun
```

`Allrun` copies the `.orig` initial fields, builds the mesh, applies `setFields`, and advances `foamRun` serially.
The tracer-transport function objects evolve the saved tracer fields during the simulation.

## Local MPI data generation

On a fresh case, use these commands instead of `Allrun`:

```bash
for field in 0/*.orig; do
    cp "$field" "${field%.orig}"
done
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
setFields > log.setFields 2>&1
decomposePar > log.decomposePar 2>&1
ranks=$(foamDictionary system/decomposeParDict -entry numberOfSubdomains -value)
mpirun -np "$ranks" foamRun -parallel > log.foamRun 2>&1
```

Run each command only after the preceding command exits successfully.
The rank count is read from the supplied decomposition dictionary; it must fit the local machine's available resources.
If adapting the MPI count, keep `numberOfSubdomains` equal to the product of the hierarchical decomposition dimensions.

## Outputs and reruns

Serial runs write raw fields in the case's time directories.
MPI runs write raw fields in `processor*/` time directories.
The commands above keep mesh and solver logs in `log.*` files.
No CSV conversion, comparison, reconstruction, or plotting workflow is included in this source package.
A completed shell command alone is not sufficient evidence of a completed production case: confirm the final saved time and the solver's normal end message in its log.

Keep prior results separately and use a fresh copy of the source case for another full run.
`Allclean` is supplied by the original case; it removes generated case data, so use it only when those outputs are intentionally disposable.

## Verification status

Input checks, launcher syntax, and documentation consistency were checked on 2026-09-23.
OpenFOAM solver execution and production-resolution simulations were not rerun during this packaging update.
See the [verification note](../../../README.md#verification) in the root README for the exact evidence boundary.
