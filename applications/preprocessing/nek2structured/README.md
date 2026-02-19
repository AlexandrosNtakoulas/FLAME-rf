# Nek To Structured

`nek2structured.py` interpolates Nek5000 field files onto a structured grid and writes HDF5 snapshots.

It supports:
- single or multiple time steps
- serial and MPI execution
- configurable structured-grid bounds and resolution
- optional 2D mesh extrusion for interpolation stability
- optional progress-variable computation from temperature

## Files

- Script: `applications/preprocessing/nek2structured/nek2structured.py`
- Config: `applications/preprocessing/nek2structured/nek2structured.yaml`

## Quick Start

1. Edit config:
```bash
nano applications/preprocessing/nek2structured/nek2structured.yaml
```

2. Run serial:
```bash
python applications/preprocessing/nek2structured/nek2structured.py
```

3. Run MPI:
```bash
mpirun -n 8 python applications/preprocessing/nek2structured/nek2structured.py
```

## Input Layout

Default input root: `DATA_BASE_DIR: "data/nek"`

Example:
```text
data/nek/
  phi0.40/
    h400x025_ref/
      premix0.f00001
      premix0.f00002
      ...
```

If `POST: true`, files are expected as:
```text
po_postPremix0.f00001
po_postPremix0.f00002
...
```

## Output Layout

Default output root: `OUTPUT_BASE_DIR: "data/fields/structured_grids"`

Example:
```text
data/fields/structured_grids/phi_0.40/h400x025_ref/
  points.hdf5
  structured_fields00290.hdf5
  structured_fields00291.hdf5
  ...
```

Output notes:
- `points.hdf5` contains structured-grid coordinates (`x`, `y`, `z`) and `nx`, `ny`, `nz` attributes.
- Interpolated files are written per timestep with zero-padded index.

## Configuration Reference

### Case and time
- `PHI` (float): equivalence ratio
- `LAT_SIZE` (string): lateral size identifier, e.g. `"025"`
- `POST` (bool): use post-processed prefix by default
- `MULTI_TIME_STEP` (bool): process range when `true`
- `TIME_STEP` (int): single timestep when `MULTI_TIME_STEP: false`
- `TIME_STEP_START` (int): start of inclusive range
- `TIME_STEP_END` (int): end of inclusive range

### Nek file naming
- `FILE_PREFIX` (string or null): custom prefix (`null` => auto from `POST`)
- `FILE_INDEX_PAD` (int): timestep zero-padding, e.g. `5` for `f00001`
- `GEOM_TIME_STEP` (int): timestep used to load mesh geometry

### Paths
- `DATA_BASE_DIR` (string): input root
- `OUTPUT_BASE_DIR` (string): output root
- `OUTPUT_FILE_STEM` (string): output snapshot prefix (default `structured_fields`)
- `POINTS_FNAME` (string): coordinates file name (default `points.hdf5`)

### Structured grid
- `X_BBOX`, `Y_BBOX`, `Z_BBOX` (2-item list): `[min, max]` bounds
- `NX`, `NY`, `NZ` (int): grid resolution
- `GRID_MODE` (string): spacing mode (default `equal`)
- `GRID_GAIN` (float): spacing gain parameter

### Interpolation settings
- `FIELDS_TO_INTERPOLATE` (list): `"all"` or explicit field names
- `POINT_INTERPOLATOR_TYPE`
- `MAX_PTS`
- `FIND_POINTS_ITERATIVE`
- `FIND_POINTS_COMM_PATTERN`
- `ELEM_PERCENT_EXPANSION`
- `GLOBAL_TREE_TYPE`
- `GLOBAL_TREE_NBINS`
- `FIND_POINTS_TOL`
- `FIND_POINTS_MAX_ITER`
- `USE_AUTOGRAD`
- `LOCAL_DATA_STRUCTURE`
- `USE_ORIENTED_BBOX`
- `WRITE_COORDS`

### 2D mesh handling
- `EXTRUDE_2D` (bool): extrude 2D mesh before interpolation
- `EXTRUDE_LZ` (int or null): override extrusion layers (`null` uses mesh order)

### Progress variable options
- `ADD_PROGRESS_VAR` (bool)
- `TEMP_FIELD_NAME` (string): source temperature field name
- `PROGRESS_FIELD_NAME` (string): output dataset name
- `PROGRESS_T_COLD_ND`, `PROGRESS_T_HOT_ND` (float or null): manual bounds
- `T_REF`, `P_REF`, `CANTERA_YAML`, `PROGRESS_FUEL`, `PROGRESS_OXIDIZER`, `PROGRESS_LOGLEVEL`: used when manual bounds are not provided

## Notes

- The script skips missing timestep files and reports them.
- For each processed timestep, output is typically `<OUTPUT_FILE_STEM><time_step>.hdf5`.
- If `ADD_PROGRESS_VAR: true`, the progress variable is appended to each output file after interpolation.
