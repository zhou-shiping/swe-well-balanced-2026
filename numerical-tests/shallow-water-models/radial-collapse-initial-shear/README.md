# Example 5.5.2: collapse with initial shear

Public package updated: 2026-09-23.
Repository-relative path: `numerical-tests/shallow-water-models/radial-collapse-initial-shear/`.

The prescribed divergence-free rotating initial shear has a nonconstant vertical profile with nonzero coefficients in both moment families.
The G-SWLME solver supports any nonnegative moment order with perfect-slip boundaries and zero explicit viscosity.
The current production settings use $N=0,1,2,3$ for the mesh and time-step study and add an $N=4$ run at $400^2$.
The initial cosine profile is unchanged; modes 1--3 use explicit coefficients and higher modes use Gauss quadrature.
The default uses second-order reconstruction, eight-point path quadrature, final time $3\,\mathrm{s}$, and CFL $0.3$.

## Local setup and trial run

Install and activate the environment using the [root installation instructions](../../../README.md#installation).
From the repository root, enter this example:

```bash
cd numerical-tests/shallow-water-models/radial-collapse-initial-shear
```

Run a small trial, with output separate from the production data:

```bash
for order in 0 1 2 3 4; do
    python run_case.py --order "$order" --nx 12 --final-time 0.001 --output-root results-trial
done
```

The trial checks execution and file writing; it does not reproduce the manuscript's resolution or final time.
For another trial, choose a new output directory or explicitly remove only the disposable trial output.

## Production data generation

Run the following from this example folder in the activated environment:

```bash
for order in 0 1 2 3; do
    for nx in 200 400 800; do
        python run_case.py --order "$order" --nx "$nx"
    done
    python run_case.py --order "$order" --nx 400 --cfl 0.15
done
python run_case.py --order 4 --nx 400 --cfl 0.3 --final-time 3
```

The driver writes one `N...` case directory per configuration under `results/`.
Each contains compressed state snapshots at $t=0,1,2,3$, CSV run diagnostics, and `metadata.json` recording the initial moment coefficients and source hashes.
Trials save only their requested final time and initial state.
The current N4 workflow consists of the single $400^2$ production run shown above; no N4 refinement or half-CFL study is implied.
Use `--help` to inspect grid, time, and output options.
The driver refuses to overwrite an existing case; select a new output root for a repeat run.
The Python drivers start from their initial conditions; they do not provide a checkpoint-resume option.
Standalone analysis, plotting, comparison, export, and verification programs are not included.

## Package status

Updated for data-generation-only distribution on 2026-09-23.
Generated data are not bundled.
See the [verification note](../../../README.md#verification) in the root README for checks performed and their limitations.
