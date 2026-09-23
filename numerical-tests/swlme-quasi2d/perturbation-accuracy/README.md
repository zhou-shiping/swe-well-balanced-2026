# Example 4: perturbation accuracy

Public package updated: 2026-09-23.
Repository-relative path: `numerical-tests/swlme-quasi2d/perturbation-accuracy/`.

Gaussian height perturbations of amplitudes $0.05$ and $0.005$ are applied to the Example 3B equilibrium.
The three methods share the same continuous initial-value problem and initial cell averages.
Default grids are $100,200,400,800$, final time is $1$, CFL is $0.25$, and the reference grids are $3200$ and $6400$.

## Local setup and trial run

Install and activate the environment using the [root installation instructions](../../../README.md#installation).
From the repository root, enter this example:

```bash
cd numerical-tests/swlme-quasi2d/perturbation-accuracy
```

Run a small trial, with output separate from the production data:

```bash
python code/run_perturbation_accuracy.py --grids 8 16 --reference 64 --reference-check 32 --amplitudes 0.05 0.005 --final-time 0.001 --output-dir results-trial
```

The trial checks execution and file writing; it does not reproduce the manuscript's resolution or final time.
For another trial, choose a new output directory or explicitly remove only the disposable trial output.

## Production data generation

Run the following from this example folder in the activated environment:

```bash
python code/run_perturbation_accuracy.py
```

The drivers save numerical states, run-time diagnostics, and configuration metadata in `results/`.
Use `--help` to inspect grid, time, and output options.
Use a separate output directory for a repeat run matrix to avoid replacing earlier outputs.
The Python drivers start from their initial conditions; they do not provide a checkpoint-resume option.
Standalone analysis, plotting, comparison, export, and verification programs are not included.

## Package status

Updated for data-generation-only distribution on 2026-09-23.
Generated data are not bundled.
See the [verification note](../../../README.md#verification) in the root README for checks performed and their limitations.
