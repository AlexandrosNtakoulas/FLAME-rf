# Model Verification

`Model_verification.ipynb` validates model behavior on extracted flame-front data and generates verification plots.

This folder also includes:
- `Markstein_lengths_calculation.ipynb`: computes Markstein-length metrics and related plots.

## Files

- Notebook: `applications/case_studies/Model_verification/Model_verification.ipynb`
- Notebook: `applications/case_studies/Model_verification/Markstein_lengths_calculation.ipynb`
- Config: `applications/case_studies/Model_verification/Model_verification.yaml`
- Global plot defaults: `applications/case_studies/plot_style.yaml`

## Quick Start

1. Edit config:
```bash
nano applications/case_studies/Model_verification/Model_verification.yaml
```

2. Open notebook:
```bash
jupyter notebook applications/case_studies/Model_verification/Model_verification.ipynb
```

3. For Markstein-length analysis:
```bash
jupyter notebook applications/case_studies/Model_verification/Markstein_lengths_calculation.ipynb
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

Default output root: `OUT_BASE_DIR: "report_figures/case_studies/Model_verification"`

Example:
```text
report_figures/case_studies/Model_verification/phi_0.4/h400x200_ref/
  t_200/
    ...
```

## Main Configuration

### Case/data selection
- `PHI` (`float`): equivalence ratio used to resolve case directories.
- `LAT_SIZE` (`str`): lateral-size tag used in folder naming.
- `POST` (`bool`): whether to load post-processed isocontour files.
- `TIME_STEPS` (`list[int]`): timesteps included in verification.
- `PROGRESS_LEVELS` (`list[float]`): progress-variable isolevels to analyze.
- `BASE_DIR` (`str`): root directory for flame-front HDF5 files.
- `MULTIPLE_RUNS` (`bool`): enable multi-run case naming logic if supported.
- `N_RUN` (`int`): run index when `MULTIPLE_RUNS` is enabled.

### Verification settings
- `CURVATURE_COLUMN` (`str`): curvature feature name used for splitting.
- `CURVATURE_LOW` (`float`): upper bound for the negative-curvature subset.
- `CURVATURE_HIGH` (`float`): lower bound for the positive-curvature subset.
- `REMOVE_OUTLIERS` (`bool`): enable IQR-based outlier removal (positive-curvature fits).
- `OUTLIER_IQR_SCALE` (`float`): aggressiveness of IQR outlier filtering.
- `C_CONSTANT` (`float`): user-defined constant used in the negative-curvature model.
- `SAVE_SPLITS` (`bool`): save curvature-split dataframes to disk.

### Output and plotting
- `OUT_BASE_DIR` (`str`): root folder for generated verification outputs.
- `FIG_SIZE` (`list[float]`): local figure-size override for this notebook.
- `COLORMAP` (`str`): colormap used for scalar-style plots.
- `SCATTER_LINEWIDTH` (`float`): marker edge width for scatter figures.
- `SD_PLOT_PERCENTILE_LOWER` (`float`): lower percentile bound for displayed `Sd` range.
- `SD_PLOT_PERCENTILE_UPPER` (`float`): upper percentile bound for displayed `Sd` range.
- Global style defaults come from `applications/case_studies/plot_style.yaml`.

## Notes

- Plot style is merged as: global defaults + local YAML overrides.
- `Markstein_lengths_calculation.ipynb` uses the same style/config merge path.
