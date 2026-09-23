# Example 3B: frictional moving equilibrium

Public package updated: 2026-09-23.
Repository-relative path: `numerical-tests/swlme-quasi2d/stationary-preservation/example3b/`.

A DOP853 reference for the full pseudo-two-dimensional stationary branch is compared with the stored equilibrium.
The standard projected Navier-slip source is active; both mean velocities and both moment families are retained.

## Local setup and trial run

Install and activate the environment using the [root installation instructions](../../../../README.md#installation).
From the repository root, enter this example:

```bash
cd numerical-tests/swlme-quasi2d/stationary-preservation/example3b
```

Run a small trial, with output separate from the production data:

```bash
python code/run_example3b.py --grids 12 --times 0 0.001 --output-dir results-trial
```

The trial checks execution and file writing; it does not reproduce the manuscript's resolution or final time.
For another trial, choose a new output directory or explicitly remove only the disposable trial output.

## Production data generation

Run the following from this example folder in the activated environment:

```bash
python code/run_example3b.py
```

The drivers save numerical states, run-time diagnostics, and configuration metadata in `results/`.
Use `--help` to inspect grid, time, and output options.
Use a separate output directory for a repeat run matrix to avoid replacing earlier outputs.
The Python drivers start from their initial conditions; they do not provide a checkpoint-resume option.
Standalone analysis, plotting, comparison, export, and verification programs are not included.

## Package status

Updated for data-generation-only distribution on 2026-09-23.
Generated data are not bundled.
See the [verification note](../../../../README.md#verification) in the root README for checks performed and their limitations.
