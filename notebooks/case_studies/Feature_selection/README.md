# Feature Selection

`Feature_selection.ipynb` performs feature-ranking and feature-subset analysis for predicting flame-related targets from isocontour datasets.

## Files

- Notebook: `notebooks/case_studies/Feature_selection/Feature_selection.ipynb`
- Config: `notebooks/case_studies/Feature_selection/Feature_selection.yaml`
- Global plot defaults: `notebooks/case_studies/plot_style.yaml`

## Quick Start

1. Edit config:
```bash
nano notebooks/case_studies/Feature_selection/Feature_selection.yaml
```

2. Open notebook:
```bash
jupyter notebook notebooks/case_studies/Feature_selection/Feature_selection.ipynb
```

## Input Layout

Default input root: `BASE_DIR: "data/isocontours"`

Example:
```text
data/isocontours/phi0.40/h400x100_ref/
  extracted_flame_front_<TIME>_iso_<ISO>.hdf5
  ...
```

## Output Layout

Default output root: `OUTPUT_BASE_DIR: "report_figures/case_studies/Feature_selection"`

Example:
```text
report_figures/case_studies/Feature_selection/phi_0.4/h400x100_ref/t_200/
  ...
```

## Main Configuration

### Case/data selection
- `PHI` (`float`): equivalence ratio used for case lookup.
- `LAT_SIZE` (`str`): lateral-size tag in the case folder.
- `POST` (`bool`): whether the notebook expects post-processed front files.
- `TIME_STEPS` (`list[int]`): snapshots used to build the dataset.
- `ISOLEVEL` (`float`): single progress-variable isolevel to load.
- `BASE_DIR` (`str`): root folder containing isocontour data.

### Target/features
- `TARGET_VAR` (`str`): regression target column.
- `TARGET_LABEL` (`str`): display label for the target in plots.
- `HIST_BINS` (`int`): number of bins in target/feature histograms.
- `CLUSTER_ON_SPATIAL` (`bool`): include coordinates (`x,y,z`) in clustering space.
- `CLUSTER_FEATURES_INCLUDE` (`list[str]`): features used to form similarity clusters.
- `MODEL_FEATURES_INCLUDE` (`list[str]`): candidate predictors used in model training.
- `FEATURES_EXCLUDE` (`list[str]`): columns to explicitly drop before selection.
- `CURVATURE_COLUMN` (`str`): feature used to define curvature bins.
- `CURVATURE_BOUNDS` (`list[float]`): bin boundaries for curvature-based splitting.

### Selection and model settings
- `MIN_CLUSTER_SAMPLES` (`int`): minimum records required per cluster.
- `RANDOM_STATE` (`int`): seed for reproducibility.
- `TEST_SIZE` (`float`): test-set fraction for train/test split.
- `MODEL_PARAMS` (`dict`): `RandomForestRegressor` hyperparameters.
- `MI_CLUSTER_THRESHOLD` (`float`): threshold used when clustering MI relationships.
- `MI_CLUSTER_NORMALIZE` (`bool`): normalize MI scores before thresholding.
- `MI_CLUSTER_LINKAGE` (`str`): linkage strategy for MI clustering.
- `MI_TOPN_PLOT` (`int`): number of top MI-ranked features to visualize.
- `TOPK_REP_PRED` (`int`): legacy shortcut for representative-predictor count.
- `K_SELECT_GLOBAL` (`int`): number of globally selected features.
- `K_SELECT_PER_CLUSTER` (`int`): number of selected features per curvature cluster.
- `BACKWARD_N_FEATURES_GLOBAL` (`int`): target feature count in global backward selection.
- `BACKWARD_N_FEATURES_CLUSTER` (`int`): target feature count in cluster-wise backward selection.
- `SFS_SCORING` (`str`): scoring metric used by sequential backward selection.
- `SFS_CV_SPLITS` (`int`): CV folds used by sequential selection.

### Output and plotting
- `OUTPUT_BASE_DIR` (`str`): root output directory for feature-selection artifacts.
- `FIG_SIZE` (`list[float]`): local figure-size override.
- `DENDRO_FIG_SIZE` (`list[float]`): dedicated figure size for dendrogram plots.
- `DENDRO_LINE_WIDTH` (`float`): branch line width in dendrogram figures.
- Global style defaults come from `notebooks/case_studies/plot_style.yaml`.

## Notes

- Global and local plotting settings are merged automatically.
