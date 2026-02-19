# FDS Decomposition Analysis

`FDS_decomposition_analysis.ipynb` analyzes source/conduction/diffusion term decomposition against target flame metrics and produces cross-plots and diagnostics.

## Files

- Notebook: `applications/case_studies/FDS_decomposition_analysis/FDS_decomposition_analysis.ipynb`
- Config: `applications/case_studies/FDS_decomposition_analysis/FDS_decomposition_analysis.yaml`
- Global plot defaults: `applications/case_studies/plot_style.yaml`

## Quick Start

1. Edit config:
```bash
nano applications/case_studies/FDS_decomposition_analysis/FDS_decomposition_analysis.yaml
```

2. Open notebook:
```bash
jupyter notebook applications/case_studies/FDS_decomposition_analysis/FDS_decomposition_analysis.ipynb
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

Default output root: `OUT_BASE_DIR: "report_figures/case_studies/FDS_decomposition_analysis"`

Example:
```text
report_figures/case_studies/FDS_decomposition_analysis/phi_0.4/h400x200_ref/
  t_200/
    ...
```

## Main Configuration

### Case/data selection
- `PHI` (`float`): equivalence ratio used to locate the case folder.
- `LAT_SIZE` (`str`): lateral-size tag in the case folder name.
- `POST` (`bool`): whether to read post-processed front files.
- `TIME_STEPS` (`list[int]`): snapshots to load.
- `PROGRESS_LEVELS` (`list[float]`): progress-variable isolevels to include.
- `BASE_DIR` (`str`): root directory containing isocontour HDF5 files.
- `MULTIPLE_RUNS` (`bool`): enable multi-run case naming if your `Case` object supports it.
- `N_RUN` (`int`): run index when `MULTIPLE_RUNS` is enabled.
- `SORET_TERM` (`bool`): toggle Soret-term-specific analysis path.
- `SORET_POSITIVE` (`bool`): optional sign convention for Soret analysis.

### Analysis setup
- `TERMS` (`list[str]`): decomposition terms to compare (e.g. source/conduction/diffusion).
- `TARGET` (`str`): target variable used for comparisons.
- `CURVATURE_COLUMN` (`str`): column used for curvature-based coloring/splits.
- `SD_COLUMN` (`str`): column used for filtering by displacement speed.
- `SD_MIN` (`float | null`): lower bound for `SD_COLUMN` (`null` disables).
- `SD_MAX` (`float | null`): upper bound for `SD_COLUMN` (`null` disables).
- `MIN_SAMPLES` (`int`): minimum row count required before plotting a subset.

### Plot controls/output
- `FIG_DPI` (`int`): figure resolution used for saved images.
- `SCATTER_S` (`float`): scatter marker size.
- `ALPHA` (`float`): scatter opacity.
- `LINEWIDTHS` (`float`): scatter marker edge width.
- `RASTERIZE` (`bool`): rasterize dense scatter layers in vector exports.
- `COLORMAP` (`str`): colormap name for scalar-colored plots.
- `COLOR_CROSS_BY_TARGET` (`bool`): color cross-plots by target values instead of a fixed color.
- `SAVE_FIGS` (`bool`): enable figure export.
- `OUT_BASE_DIR` (`str`): root output directory for generated figures.
- `PLOT_MARGIN_LEFT` (`float`): left subplot margin.
- `PLOT_MARGIN_RIGHT` (`float`): right subplot margin.
- `PLOT_MARGIN_BOTTOM` (`float`): bottom subplot margin.
- `PLOT_MARGIN_TOP` (`float`): top subplot margin.
- `PLOT_WSPACE` (`float`): subplot horizontal spacing.
- `SCATTER_CBAR_FRACTION` (`float`): colorbar size fraction for scatter plots.
- `SCATTER_CBAR_PAD` (`float`): colorbar spacing from axes.
- `SAVE_PAD_INCHES` (`float`): extra padding around saved figures.
- `SAVE_TIGHT_BBOX` (`bool`): export using tight bounding box when true.
- Global style defaults come from `applications/case_studies/plot_style.yaml`.

## Notes

- Plot style is merged from global defaults plus local overrides.
