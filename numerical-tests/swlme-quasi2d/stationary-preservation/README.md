# Example 3: moving-equilibrium preservation

Public package updated: 2026-09-23.
Repository-relative path: `numerical-tests/swlme-quasi2d/stationary-preservation/`.

Example 3A is a frictionless one-dimensional moving branch; Example 3B is a frictional pseudo-two-dimensional branch with both velocity and moment families.
The shared numerical core is `code/quasi2d_wb.py`.
Both compare standard HLL, WB1, and WB2 on $N_x=100,200,400,800$ through $t=10$.

## Data generation

See the subfolder READMEs for the independent data-generation commands and production run matrices.

## Package status

Updated for data-generation-only distribution on 2026-09-23.
Generated data are not bundled.
See the [verification note](../../../README.md#verification) in the root README for checks performed and their limitations.
