"""Extract full-field data from Nek5000 .f* files and write to HDF5.

This script is compatible with the **newest** flamekit.datasets.py API (SEMDataset +
extract_full_field_hdf5 in datasets_complete.py / your updated datasets.py).

Key behavior:
  - Build SEMDataset once (mesh + Coef once)
  - Reload only fields per timestep
  - Serial or MPI (pySEMTools distributes the mesh/fields across ranks)
  - Output naming:
      * size==1  -> <field_path>.h5
      * size>1   -> <field_path>_rank####.h5   (one file per rank)

Run examples:
  - Serial:
      PYTHONPATH=. python notebooks/preprocessing/extract_fields/extract_fields_updated.py
  - MPI:
      PYTHONPATH=. mpirun -n 8 python notebooks/preprocessing/extract_fields/extract_fields_updated.py

Config file:
  <PROJECT_ROOT>/notebooks/preprocessing/extract_fields/extract_fields.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from mpi4py import MPI

from flamekit.io_fronts import Case, folder
from flamekit.io_fields import make_case_with_base_dir, field_path
from flamekit.datasets import SEMDataset
import pandas as pd
import numpy as np

def _find_project_root(start: Path) -> Path:
    """Find repo root by walking up until 'notebooks' directory exists."""
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "notebooks").is_dir():
            return p
    return Path.cwd().resolve()



def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    project_root = _find_project_root(Path(__file__).resolve().parent)
    os.chdir(str(project_root))

    config_path = project_root / "notebooks" / "preprocessing" / "extract_fields" / "extract_fields.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_base_dir = project_root / Path(cfg["DATA_BASE_DIR"])
    output_base_dir = project_root / Path(cfg["OUTPUT_BASE_DIR"])
    post = bool(cfg["POST"])
    file_name = "po_postPremix" if post else "premix"
    hdf_format = str(cfg.get("HDF_FORMAT", "fixed"))

    # time steps
    time_step = int(cfg["TIME_STEP"])
    multi_time_step = bool(cfg["MULTI_TIME_STEP"])
    time_step_start = int(cfg["TIME_STEP_START"])
    time_step_end = int(cfg["TIME_STEP_END"])
    
    # case-identifier
    phi = float(cfg["PHI"])
    lat_size = str(cfg["LAT_SIZE"])

    # scalars
    scalars = list(cfg["SCALARS"])

    # feature toggles
    compute_vel_jacobian = bool(cfg["COMP_VEL_JACOBIAN"])
    compute_vel_hessian = bool(cfg["COMP_VEL_HESSIAN"])
    compute_T_grad = bool(cfg["COMP_T_GRAD"])
    compute_curv_grad = bool(cfg["COMP_CURV_GRAD"])
    compute_local_vel_jacobian = bool(cfg["COMP_LOCAL_VEL_JACOBIAN"])
    compute_reaction_rates = bool(cfg.get("COMP_REACTION_RATES", False))

    # Cantera inputs
    T_REF = float(cfg["T_REF"])
    P_REF = float(cfg["P_REF"])
    CANTERA_YAML = project_root / Path(cfg["CANTERA_YAML"])
    CANTERA_INPUTS = [str(CANTERA_YAML), None, T_REF, P_REF]  # [cantera_file, species_list, t_ref, p_ref]

    # Determine timesteps to process
    if multi_time_step:
        ts_list = list(range(time_step_start, time_step_end + 1))
    else:
        ts_list = [time_step]

    # Build SEMDataset once (mesh + coef once)
    init_ts = int(ts_list[0])
    case0 = Case(base_dir=output_base_dir, phi=phi, lat_size=lat_size, time_step=init_ts, post=post)
    data_case0 = make_case_with_base_dir(case0, data_base_dir)
    data_folder0 = folder(data_case0)

    ds = SEMDataset(
        folder_name=str(data_folder0),
        file_name=file_name,
        time_step=init_ts,
        comm=comm,
        scalar_names=scalars,
    )

    # Process timesteps
    for ts in ts_list:
        ts = int(ts)
        if rank == 0:
            print(f"=== Processing TIME_STEP={ts} (MPI size={size}) ===")

        case = Case(base_dir=output_base_dir, phi=phi, lat_size=lat_size, time_step=ts, post=post)
        out_case = make_case_with_base_dir(case, output_base_dir)
        out_hdf_path = field_path(out_case)

        # Local read
        ds.reload_timestep(case.time_step)

        scalar_subset = None

        point_data = ds.build_point_data_dict(
            cantera_inputs=CANTERA_INPUTS,
            phi=phi,
            scalar_subset=scalar_subset,
            compute_progress_var=True,
            compute_vel_jacobian=compute_vel_jacobian,
            compute_vel_hessian=compute_vel_hessian,
            compute_T_grad=compute_T_grad,
            compute_curv_grad=compute_curv_grad,
            compute_local_vel_jacobian=compute_local_vel_jacobian,
            compute_reaction_rates=compute_reaction_rates,
        )
        # Convert to flat columns for pandas (rank-local)
        data = {
            "x": np.asarray(ds.x).ravel(order="C"),
            "y": np.asarray(ds.y).ravel(order="C"),
            "z": np.asarray(ds.z).ravel(order="C"),
        }

        n = data["x"].size

        for k, arr in point_data.items():
            v = np.asarray(arr).ravel(order="C")
            if v.size != n:
                raise ValueError(f"Column {k!r} has size {v.size}, expected {n}. Shape was {np.asarray(arr).shape}.")
            data[k] = v

        df = pd.DataFrame(data)

        # Output file per rank
        out_file = out_hdf_path if size == 1 else out_hdf_path.with_name(f"{out_hdf_path.stem}_rank{rank:04d}{out_hdf_path.suffix}")
        df.to_hdf(out_file, key="field", mode="w", format=hdf_format, index=False)

        if rank == 0:
            print(f"[rank0] Wrote: {out_file}")

    comm.Barrier()


if __name__ == "__main__":
    main()
