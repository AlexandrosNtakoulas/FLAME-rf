# CNN

`CNN.ipynb` trains and evaluates a convolutional model on structured-grid HDF5 snapshots for short-horizon temporal prediction.

## Files

- Notebook: `notebooks/case_studies/CNN/CNN.ipynb`
- Config: `notebooks/case_studies/CNN/CNN.yaml`
- Global plot defaults: `notebooks/case_studies/plot_style.yaml`

## Quick Start

1. Edit config:
```bash
nano notebooks/case_studies/CNN/CNN.yaml
```

2. Open notebook:
```bash
jupyter notebook notebooks/case_studies/CNN/CNN.ipynb
```

## Input Layout

Default input root: `BASE_DIR: "data/fields/structured_grids"`

Example:
```text
data/fields/structured_grids/phi_0.40/h400x025_ref/
  points.hdf5
  structured_fields00200.hdf5
  structured_fields00201.hdf5
  ...
```

## Output Layout

Prediction output root:
`PRED_OUT_DIR: "data/fields/cnn_predictions"`

Training/evaluation plots are saved under the notebook run output directories.

## Main Configuration

### Data/time selection
- `TIME_STEP_START` (`int`): first training timestep (inclusive).
- `TIME_STEP_END` (`int`): last training timestep (inclusive).
- `PRED_TIME_STEP_START` (`int | null`): first optional prediction-evaluation timestep.
- `PRED_TIME_STEP_END` (`int | null`): last optional prediction-evaluation timestep.
- `PHI` (`float`): equivalence ratio used to locate case data.
- `LAT_SIZE` (`str`): lateral-size case tag.
- `BASE_DIR` (`str`): root directory containing structured-grid cases.
- `CASE_DIR` (`str | null`): optional direct case-path override (bypasses `PHI`/`LAT_SIZE` path logic).
- `OUTPUT_FILE_STEM` (`str`): filename stem for structured snapshots.
- `FILE_INDEX_PAD` (`int`): zero-padding width for timestep indices.
- `POINTS_FNAME` (`str`): structured-grid coordinate file name.
- `FIELD_NAME` (`str`): dataset key to model from each HDF5 snapshot.
- `USE_TIME_ATTR` (`bool`): sort snapshots by HDF5 time attribute when available.

### Training/model settings
- `DEVICE` (`str`): training backend (`cpu` or `cuda`).
- `REMOVE_MEAN` (`bool`): subtract dataset mean prior to training.
- `EPS` (`float`): numerical epsilon used in normalization operations.
- `HISTORY` (`int`): number of past frames used as model input.
- `TRAIN_FRACTION` (`float`): fraction of samples allocated to training.
- `EPOCHS` (`int`): number of training epochs.
- `BATCH_SIZE` (`int`): mini-batch size.
- `LR` (`float`): optimizer learning rate.
- `WEIGHT_DECAY` (`float`): L2 regularization coefficient.
- `DROPOUT` (`float`): dropout probability.
- `HIDDEN_CHANNELS` (`list[int]`): channel sizes across CNN blocks.
- `KERNEL_SIZE` (`int`): convolution kernel size.
- `PREDICT_DELTA` (`bool`): predict frame increments (`delta`) instead of absolute values.
- `LOG_EVERY` (`int`): logging interval (steps or batches, notebook-dependent).
- `SEED` (`int`): random seed for reproducibility.

### Study/prediction options
- `HISTORY_CANDIDATES` (`list[int]`): candidate history lengths for sweep experiments.
- `HISTORY_STUDY_EPOCHS` (`int`): training epochs used in history-length studies.
- `PRED_WINDOW` (`int`): autoregressive prediction horizon length.
- `PRED_START_INDEX` (`int | null`): optional start index for rolling prediction analysis.
- `PRED_OUT_DIR` (`str`): output directory for saved prediction snapshots/files.

### Plot/isocontour options
- `PLOT_SLICE_INDEX` (`int | null`): optional z-slice index to plot for 3D arrays.
- `ISO_PROGRESS_FIELD_NAME` (`str`): field used for contour extraction/comparison.
- `ISO_PROGRESS_VALUE` (`float`): contour threshold value.
- `ISO_X_PAD_FRAC` (`float`): x-padding around contour domain in plots.
- `XLIM` (`list[float] | null`): optional x-axis zoom bounds (`null` disables).
- `YLIM` (`list[float] | null`): optional y-axis zoom bounds (`null` disables).
- Global style defaults come from `notebooks/case_studies/plot_style.yaml`.

## Notes

- This notebook expects structured-grid HDF5 input from preprocessing (`nek2structured` workflow).
