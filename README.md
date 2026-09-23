# Two-dimensional shallow water linearized moment equations

Data-generation source code for **Two-Dimensional Shallow Water Linearized Moment Equations: Hyperbolicity and Well-Balanced Schemes**, by Shiping Zhou, Juntao Huang, and Andrew J. Christlieb.

Prepared for independent release: **2026-09-23**.
Repository-relative path: `.`.
Intended repository: [swe-well-balanced-2026](https://github.com/zhou-shiping/swe-well-balanced-2026).
This repository is prepared locally for review; public GitHub publication, a versioned release, and a Zenodo DOI are pending.

## Scope and manuscript mapping

The numerical code implements the globally hyperbolic two-dimensional G-SWLME, quasi-two-dimensional well-balanced schemes, and the manuscript's numerical comparisons.
The frictional examples use the standard projected Navier-slip source.
The well-balanced claims concern prescribed, fully wet, one-coordinate-dependent equilibria with the full moment state.

| Manuscript section | Experiment | Source folder |
|---|---|---|
| 5.1, Example 1A | Smooth two-dimensional same-equation convergence | [example1a](numerical-tests/shallow-water-models/moment-benchmark-2d/example1a/) |
| 5.1, Example 1B | Comparison with a vertically resolved hydrostatic parent system | [example1b](numerical-tests/shallow-water-models/moment-benchmark-2d/example1b/) |
| 5.2 | Lake-at-rest preservation | [lake-at-rest](numerical-tests/swlme-quasi2d/lake-at-rest/) |
| 5.3 | Frictionless and frictional moving-equilibrium preservation | [stationary-preservation](numerical-tests/swlme-quasi2d/stationary-preservation/) |
| 5.4 | Perturbation accuracy and reference/time-step sensitivity | [perturbation-accuracy](numerical-tests/swlme-quasi2d/perturbation-accuracy/) |
| 5.5.1 | Perfect-slip radial collapse from rest | [moment model](numerical-tests/shallow-water-models/radial-collapse-perfect-slip/), [OpenFOAM](numerical-tests/openfoam/radial-collapse-perfect-slip/) |
| 5.5.2 | Perfect-slip radial collapse with initial shear | [moment model](numerical-tests/shallow-water-models/radial-collapse-initial-shear/), [OpenFOAM](numerical-tests/openfoam/radial-collapse-initial-shear/) |

Included: numerical solvers, simulation drivers, state-saving and run-time diagnostic routines, OpenFOAM input dictionaries and initializers, and execution instructions.
Both OpenFOAM examples include explicit N200, N400, and N800 production input folders.
Postprocessing, plotting, comparison, export, and standalone verification programs are excluded.
Generated simulation data, figures, logs, cluster submission scripts, manuscript files, and private research notes are excluded.

## Installation

Use Python 3.11 or newer. The commands below use a macOS/Linux shell (or a configured Linux shell on Windows).
Open a terminal in the repository root and create an isolated Python environment outside the source folder:

```bash
python3 -m venv "$HOME/.venvs/swe-well-balanced-2026"
source "$HOME/.venvs/swe-well-balanced-2026/bin/activate"
python -m pip install -r requirements.txt
```

Activate this environment in each new terminal before running Python examples.
All examples run with ordinary `python` commands; no scheduler, module system, or cluster account is needed.
The Python solvers run on the local CPU and require NumPy; the stationary-reference calculations additionally require SciPy.

The local verification environment uses Python 3.11.11, NumPy 2.3.5, and SciPy 1.17.0.
The OpenFOAM cases require a separately installed OpenFOAM Foundation environment with `foamRun` and the incompressible VOF solver; the rest-case dictionaries identify version 12, and the initial-shear setup was prepared for version 11.
OpenFOAM runtime compatibility has not been tested during this packaging task.

## Quick start

For a small numerical run, enter Example 1A and use:

```bash
cd numerical-tests/shallow-water-models/moment-benchmark-2d/example1a
python run_case.py --order 2 --nx 16 --final-time 0.01
```

Each example README gives its exact working directory, a small trial command where applicable, production commands matching the current research launch settings, output locations, and rerun behavior.
Use `--help` on drivers with command-line options before launching large calculations.
Default production cases, especially fine reference grids and three-dimensional OpenFOAM meshes, can require substantial compute resources.

## Current initial-shear settings

The initial-shear moment generator supports nonnegative moment orders.
The current run matrix includes $N=0,1,2,3$ on $200^2$, $400^2$, and $800^2$ grids with matched $400^2$ half-CFL controls, plus $N=4$ on $400^2$ at CFL $0.3$.
See its [local-run guide](numerical-tests/shallow-water-models/radial-collapse-initial-shear/README.md).
The three OpenFOAM resolution folders and their physical setup remain unchanged.

## Outputs

Generated outputs are written under the example folders, in `results/` for the Python generators and case time directories for OpenFOAM.
The root `.gitignore` is copied byte-for-byte from `swe-public`.
Additional output exclusions in [numerical-tests/.gitignore](numerical-tests/.gitignore) keep generated artifacts out of this source repository.

The source package generates numerical data but does not include the manuscript's original output data or the postprocessing used to make its tables and figures.
Any future supporting-data archive must be described separately.
The vertically resolved parent calculation is not an exact componentwise reference, and the OpenFOAM comparisons do not establish full reference convergence.

## Verification

On 2026-09-23, all eight documented Python trial workflows passed (13 driver runs), and Python syntax, shell syntax, and OpenFOAM input-consistency checks passed.
OpenFOAM solver execution and full production simulations were not rerun; these small-run checks do not constitute new production-resolution or scientific-validation results.

## Citation and archival release

Author and software metadata are provided in [CITATION.cff](CITATION.cff).
A software DOI has not yet been assigned.
After review, publish the GitHub repository, connect it to Zenodo, and create a versioned release matching the manuscript.
Record the resulting version-specific DOI and release identifier here and in the manuscript before submission.
The software archive associated with the authors' earlier paper belongs to that earlier work and should not be used as this repository's DOI.

## License

MIT License; see [LICENSE](LICENSE).
Public scholarly authorship and copyright attribution are retained.
