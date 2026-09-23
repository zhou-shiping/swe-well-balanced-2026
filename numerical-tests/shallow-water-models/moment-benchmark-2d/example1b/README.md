# Example 1B: vertically resolved comparison

Public package updated: 2026-09-23.
Repository-relative path: `numerical-tests/shallow-water-models/moment-benchmark-2d/example1b/`.

The inviscid mapped hydrostatic parent system uses layer-centered horizontal momenta and a continuity-compatible vertical mass flux with closed endpoints.
The local G-SWLME solver uses the same initial physical profile truncated to orders $N=0,1,2$.

## Local setup and trial run

Install and activate the environment using the [root installation instructions](../../../../README.md#installation).
From the repository root, enter this example:

```bash
cd numerical-tests/shallow-water-models/moment-benchmark-2d/example1b
```

Run a small trial, with output separate from the production data:

```bash
python run_moment.py --order 2 --nx 12 --final-time 0.001 --output-directory results-trial
python run_reference.py --nx 12 --nz 8 --final-time 0.001 --output-directory results-trial
```

The trial checks execution and file writing; it does not reproduce the manuscript's resolution or final time.
For another trial, choose a new output directory or explicitly remove only the disposable trial output.

## Production data generation

Run the following from this example folder in the activated environment:

```bash
for order in 0 1 2; do
    for nx in 128 256; do
        python run_moment.py --order "$order" --nx "$nx" --final-time 0.1 --cfl 0.3
    done
done
for nx in 32 64 128 256; do
    python run_reference.py --nx "$nx" --nz 64 --final-time 0.1 --cfl 0.2
done
for nz in 16 32 128; do
    python run_reference.py --nx 128 --nz "$nz" --final-time 0.1 --cfl 0.2
done
for cfl in 0.1 0.05; do
    python run_reference.py --nx 128 --nz 64 --final-time 0.1 --cfl "$cfl"
done
python run_reference.py --nx 256 --nz 128 --final-time 0.1 --cfl 0.1
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
