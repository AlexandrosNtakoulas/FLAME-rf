# DMD

`DMD.ipynb` runs Dynamic Mode Decomposition on extracted field snapshots and generates temporal/modal diagnostics and spatial mode plots.

This folder also includes:
- `Flame_evolution.ipynb`: isocontour evolution plotting utility using DMD case configuration.

## Files

- Notebook: `notebooks/case_studies/DMD/DMD.ipynb`
- Notebook: `notebooks/case_studies/DMD/Flame_evolution.ipynb`
- Config: `notebooks/case_studies/DMD/DMD.yaml`
- Global plot defaults: `notebooks/case_studies/plot_style.yaml`

## Quick Start

1. Edit config:
```bash
nano notebooks/case_studies/DMD/DMD.yaml
```

2. Open DMD notebook:
```bash
jupyter notebook notebooks/case_studies/DMD/DMD.ipynb
```

3. Open flame-evolution notebook:
```bash
jupyter notebook notebooks/case_studies/DMD/Flame_evolution.ipynb
```

## Input Layout

Default input root: `BASE_DIR: "data/fields"`

Example:
```text
data/fields/unstructured/phi0.40/h400x025_ref/
  extracted_field_<TIME>.hdf5
  ...
```

## Output Layout

Default output root: `OUT_DIR: "report_figures/case_studies/DMD"`

Example:
```text
report_figures/case_studies/DMD/phi_0.4/h400x025_ref/t_270_to_290/
  ...
```

## Main Configuration

### Case/time selection
- `TIME_STEP_START` (`int`): first timestep in the DMD snapshot window (inclusive).
- `TIME_STEP_END` (`int`): last timestep in the DMD snapshot window (inclusive).
- `PHI` (`float`): equivalence ratio used to locate field files.
- `LAT_SIZE` (`str`): lateral-size case tag.
- `POST` (`bool`): whether to read post-processed field naming.
- `BASE_DIR` (`str`): root directory for extracted field HDF5 files.

### DMD setup
- `VAR_NAME` (`str`): field column used to build the snapshot state vector.
- `SORT_COLS` (`list[str]`): coordinate columns used for deterministic point ordering.
- `COORD_TOL` (`float`): tolerance for coordinate-equality checks (`0` disables tolerance).
- `REMOVE_MEAN` (`bool`): subtract temporal mean before DMD when true.
- `X_THRESHOLD` (`float`): keep only points with `x > X_THRESHOLD`.
- `DMD_SVD_RANK` (`int | null`): optional rank truncation for DMD (`null` = full).
- `DT` (`float | null`): physical timestep spacing for frequency/time-scale interpretation.
- `MODE_IMAG_TOL` (`float`): tolerance for classifying near-real eigenvalues.
- `FREQ_TOL` (`float`): tolerance for small-frequency handling.

### Plot/output settings
- `OUT_DIR` (`str`): root directory for DMD plots and exports.
- `N_MODES_TO_PLOT` (`int`): number of leading modes to visualize.
- `NORMALIZE_MODE_FOR_PLOT` (`bool`): normalize mode amplitudes for display.
- `ISO_PROGRESS_VAR_NAME` (`str`): scalar used for isocontour overlays.
- `ISO_PROGRESS_VALUE` (`float`): contour level used in isocontour overlays.
- `ISO_X_PAD_FRAC` (`float`): x-padding fraction around plotted contour bounds.
- `CONTOUR_LEVELS_FILLED` (`int`): number of filled contour levels.
- `CONTOUR_LEVELS_LINES` (`int`): number of line contour levels.
- `MODE_CONTOUR_LEVELS_FILLED` (`int`): filled levels specifically for mode plots.
- `COORD_X` (`str`): x-coordinate column name for plotting.
- `COORD_Y` (`str`): y-coordinate column name for plotting.
- `XLIM` (`list[float] | null`): optional x-axis zoom bounds (`null` disables).
- `YLIM` (`list[float] | null`): optional y-axis zoom bounds (`null` disables).
- `COLORMAP` (`str`): colormap used in contour-style visualizations.
- `HISTORY_SWEEP_CMAP_MIN` (`float`): lower colormap clipping bound for sweep plots.
- `HISTORY_SWEEP_CMAP_MAX` (`float`): upper colormap clipping bound for sweep plots.
- Global style defaults come from `notebooks/case_studies/plot_style.yaml`.

## Notes

- `Flame_evolution.ipynb` reads `DMD.yaml` and uses the same merged global plot style.
