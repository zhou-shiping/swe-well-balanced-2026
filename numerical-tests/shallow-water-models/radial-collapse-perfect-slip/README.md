# Example 5.5.1: collapse from rest

Public package updated: 2026-09-23.
Repository-relative path: `numerical-tests/shallow-water-models/radial-collapse-perfect-slip/`.

The zero-moment invariant G-SWLME solution gives the shallow-water limit over a perfect-slip bed.
Production comparisons use $200^2$, $400^2$, and $800^2$ horizontal meshes, final time $3\,\mathrm{s}$, and CFL $0.4$.

## Local setup and trial run

Install and activate the environment using the [root installation instructions](../../../README.md#installation).
From the repository root, enter this example:

```bash
cd numerical-tests/shallow-water-models/radial-collapse-perfect-slip
```

Run a small trial, with output separate from the production data:

```bash
python run_case.py --nx 12 --order 0 --final-time 0.001 --results-directory results-trial
```

The trial checks execution and file writing; it does not reproduce the manuscript's resolution or final time.
For another trial, choose a new output directory or explicitly remove only the disposable trial output.

## Production data generation

Run the following from this example folder in the activated environment:

```bash
for nx in 200 400 800; do
    python run_case.py --nx "$nx" --order 0
done
for order in 1 2; do
    python run_case.py --nx 400 --order "$order"
done
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
