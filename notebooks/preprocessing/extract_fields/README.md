# Extract Fields

`extract_fields.py` reads Nek5000 `.f*` field files and exports processed data for analysis.

It supports:
- single or multiple time steps
- serial and MPI execution
- optional subdomain cropping
- optional derived quantities (gradients, transport, reaction rates, etc.)

## Files

- Script: `notebooks/preprocessing/extract_fields/extract_fields.py`
- Config: `notebooks/preprocessing/extract_fields/extract_fields.yaml`

## Quick Start

1. Edit config:
```bash
nano notebooks/preprocessing/extract_fields/extract_fields.yaml
```

2. Run serial:
```bash
python notebooks/preprocessing/extract_fields/extract_fields.py
```

3. Run MPI:
```bash
mpirun -n 8 python notebooks/preprocessing/extract_fields/extract_fields.py
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

Default output root: `OUTPUT_BASE_DIR: "data/fields/unstructured"`

Example:
```text
data/fields/unstructured/phi0.40/h400x025_ref/
  extracted_field_00001.hdf5            # serial
  extracted_field_00001_rank0000.hdf5   # MPI
  extracted_field_00001_rank0001.hdf5
  ...
```

Augmented Nek files may also be written under:
`PROCESSED_NEK_BASE_DIR` (default: `data/processed_nek`).

## Configuration Reference

### Case and time
- `PHI` (float): equivalence ratio
- `LAT_SIZE` (string): lateral size identifier, e.g. `"025"`
- `TIME_STEP` (int): single time step when `MULTI_TIME_STEP: false`
- `MULTI_TIME_STEP` (bool): process range when `true`
- `TIME_STEP_START` (int): start of inclusive range
- `TIME_STEP_END` (int): end of inclusive range
- `POST` (bool): use `po_postPremix0.f*` naming when true

### Paths and format
- `DATA_BASE_DIR` (string): input root
- `OUTPUT_BASE_DIR` (string): output root for HDF5
- `PROCESSED_NEK_BASE_DIR` (string, optional): output root for augmented Nek files
- `HDF_FORMAT` (string): `"fixed"` or `"table"`

### Subdomain options
- `KEEP_SUBDOMAIN` (bool)
- `X_LIMITS` (tuple): `(min, max)` with `None` allowed
- `Y_LIMITS` (tuple): `(min, max)` with `None` allowed
- `Z_LIMITS` (tuple): `(min, max)` with `None` allowed

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
- `SCALARS` (list): species/scalars read from field files
- `T_REF` (float): reference temperature
- `P_REF` (float): reference pressure
- `CANTERA_YAML` (string): mechanism path, e.g. `FLAME/chemical_mech/BurkeH2.yaml`

## Notes

- In MPI mode, each rank writes a separate HDF5 file.
- `"fixed"` format is usually faster; `"table"` is more flexible for table-style operations.
