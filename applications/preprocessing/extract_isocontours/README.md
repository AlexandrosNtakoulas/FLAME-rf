# Extract Isocontours

`extract_isocontours.py` extracts flame-front isocontours from Nek5000 field data and writes one HDF5 file per `(time step, iso level)`.

It supports:
- single or multiple time steps
- serial and MPI execution
- configurable progress-variable isolevels
- optional derived quantities (gradients, transport, reaction rates, etc.)

## Files

- Script: `applications/preprocessing/extract_isocontours/extract_isocontours.py`
- Config: `applications/preprocessing/extract_isocontours/extract_isocontours.yaml`

## Quick Start

1. Edit config:
```bash
nano applications/preprocessing/extract_isocontours/extract_isocontours.yaml
```

2. Run serial:
```bash
python applications/preprocessing/extract_isocontours/extract_isocontours.py
```

3. Run MPI:
```bash
mpirun -n 8 python applications/preprocessing/extract_isocontours/extract_isocontours.py
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

Default output root: `ISOCONTOUR_BASE_DIR: "data/isocontours"`

Example:
```text
data/isocontours/phi0.40/h400x025_ref/
  extracted_flame_front_00150_iso_0.600.hdf5
  extracted_flame_front_00151_iso_0.600.hdf5
  ...
```

Each file contains one table under key `front`.

## Configuration Reference

### Case and time
- `PHI` (float): equivalence ratio
- `LAT_SIZE` (string): lateral size identifier, e.g. `"025"`
- `TIME_STEP` (int): single time step when `MULTI_TIME_STEP: false`
- `MULTI_TIME_STEP` (bool): process range when `true`
- `TIME_STEP_START` (int): start of inclusive range
- `TIME_STEP_END` (int): end of inclusive range
- `POST` (bool): use `po_postPremix0.f*` naming when true

### Paths
- `DATA_BASE_DIR` (string): input root
- `ISOCONTOUR_BASE_DIR` (string): output root for front files

### Isocontour control
- `PROGRESS_LEVELS` (list[float]): progress-variable iso levels (typically in `[0, 1]`)

### Compute toggles
- `COMP_VEL_JACOBIAN`
- `COMP_VEL_HESSIAN`
- `COMP_REACTION_RATES`
- `COMP_TRANSPORT`
- `COMP_T_GRAD`
- `COMP_CURV_GRAD`
- `COMP_LOCAL_VEL_JACOBIAN`
- `COMP_DSSUM_DERIVS`

### Scalars and chemistry
- `SCALARS` (list): species/scalars loaded from field files
- `T_REF` (float): reference temperature
- `P_REF` (float): reference pressure
- `CANTERA_YAML` (string): mechanism path, e.g. `FLAME/chemical_mech/BurkeH2.yaml`

## Notes

- In MPI mode, all ranks compute local contour pieces; rank 0 gathers and writes a single file per `(time step, iso)`.
- The script computes `progress_var` and contours that field using each value in `PROGRESS_LEVELS`.
