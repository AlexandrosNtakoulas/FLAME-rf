# Tests Case Study

`tests.ipynb` is a lightweight case-study notebook for split-based evaluation and uncertainty-related diagnostics.

## Files

- Notebook: `notebooks/case_studies/tests/tests.ipynb`
- Config: `notebooks/case_studies/tests/tests.yaml`
- Global plot defaults: `notebooks/case_studies/plot_style.yaml`

## Quick Start

1. Edit config:
```bash
nano notebooks/case_studies/tests/tests.yaml
```

2. Open notebook:
```bash
jupyter notebook notebooks/case_studies/tests/tests.ipynb
```

## Input Layout

Default input root: `BASE_DIR: "data/isocontours"`

Example:
```text
data/isocontours/phi0.40/h400x200_ref/
  extracted_flame_front_<TIME>_iso_<ISO>.hdf5
  ...
```

## Output Layout

Default output root: `OUT_BASE_DIR: "report_figures/case_studies/tests"`

Example:
```text
report_figures/case_studies/tests/phi_0.4/h400x200_ref/t_200/
  ...
```

## Main Configuration

### Case/data selection
- `PHI` (`float`): equivalence ratio used to locate case directories.
- `LAT_SIZE` (`str`): lateral-size tag in folder naming.
- `POST` (`bool`): whether to load post-processed isocontour file naming.
- `TIME_STEPS` (`list[int]`): timesteps included in the analysis.
- `PROGRESS_LEVELS` (`list[float]`): isolevels to load.
- `BASE_DIR` (`str`): root folder containing isocontour HDF5 data.
- `MULTIPLE_RUNS` (`bool`): enable multi-run case naming when supported.
- `N_RUN` (`int`): run index when `MULTIPLE_RUNS` is enabled.

### Model and uncertainty settings
- `CURVATURE_COLUMN` (`str`): curvature column used for split logic.
- `CURVATURE_LOW` (`float`): upper bound for the negative-curvature subset.
- `CURVATURE_HIGH` (`float`): lower bound for the positive-curvature subset.
- `C_CONSTANT` (`float`): user-defined constant for model equations.
- `N_POST_SAMPLES` (`int`): number of posterior samples for uncertainty estimation.
- `PRIOR_ALPHA` (`float`): prior precision/hyperparameter for Bayesian-style updates.
- `POSTERIOR_SEED` (`int`): RNG seed for posterior sampling reproducibility.
- `BAND_SIGMA` (`float`): uncertainty-band multiplier.
- `MAX_POINTS_FOR_BAND` (`int`): cap on points used when computing uncertainty bands.

### Output and plotting
- `SAVE_SPLITS` (`bool`): save intermediate split datasets to files.
- `OUT_BASE_DIR` (`str`): root directory for notebook outputs.
- `PLOT_FONT_SIZE` (`int`): local font-size override for this notebook.
- Global style defaults come from `notebooks/case_studies/plot_style.yaml`.

## Notes

- Plot style is merged from global defaults and local notebook overrides.
