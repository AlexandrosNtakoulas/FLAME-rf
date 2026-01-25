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
import ast
from pathlib import Path
from typing import List, Iterable, Tuple

import yaml
from mpi4py import MPI

from flamekit.io_fronts import Case, folder
from flamekit.io_fields import make_case_with_base_dir, field_path
from flamekit.datasets import SEMDataset
import pandas as pd
import numpy as np
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.io.ppymech.neksuite import pynekwrite
from pysemtools.interpolation.interpolator import (
    get_bbox_from_coordinates,
    get_bbox_centroids_and_max_dist,
)

def _find_project_root(start: Path) -> Path:
    """Find repo root by walking up until 'notebooks' directory exists."""
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "notebooks").is_dir():
            return p
    return Path.cwd().resolve()


def _parse_limits(value: object) -> Tuple[object, object]:
    if value is None:
        return (None, None)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (value[0], value[1])
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return (None, None)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
            return (parsed[0], parsed[1])
    return (None, None)


def _normalize_limits(value: object) -> Tuple[float | None, float | None]:
    a, b = _parse_limits(value)

    def _to_float(x: object) -> float | None:
        if x is None:
            return None
        if isinstance(x, str) and x.strip().lower() == "none":
            return None
        return float(x)

    return (_to_float(a), _to_float(b))


def _select_elements_in_bounds(
    msh: Mesh,
    x_limits: Tuple[float | None, float | None],
    y_limits: Tuple[float | None, float | None],
    z_limits: Tuple[float | None, float | None],
) -> np.ndarray:
    xmin, xmax = x_limits
    ymin, ymax = y_limits
    zmin, zmax = z_limits

    if xmin is None and xmax is None and ymin is None and ymax is None and zmin is None and zmax is None:
        return np.arange(msh.nelv, dtype=int)

    bbox = get_bbox_from_coordinates(msh.x, msh.y, msh.z)

    cond = np.ones((bbox.shape[0],), dtype=bool)
    if xmin is not None:
        cond &= bbox[:, 0] >= xmin
    if xmax is not None:
        cond &= bbox[:, 1] <= xmax
    if ymin is not None:
        cond &= bbox[:, 2] >= ymin
    if ymax is not None:
        cond &= bbox[:, 3] <= ymax
    if zmin is not None:
        cond &= bbox[:, 4] >= zmin
    if zmax is not None:
        cond &= bbox[:, 5] <= zmax

    return np.where(cond)[0]


def _subset_elements(arr: np.ndarray, elem_idx: np.ndarray | None) -> np.ndarray:
    if elem_idx is None:
        return np.asarray(arr)
    return np.asarray(arr)[elem_idx, ...]


def _base_scalar_names(fld: FieldRegistry, fallback: Iterable[str]) -> List[str]:
    scal = fld.fields.get("scal", []) if fld is not None else []
    if len(fallback) == len(scal):
        return list(fallback)
    return [f"s{i}" for i in range(len(scal))]



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

    # Subdomain
    subdomain = bool(cfg.get("KEEP_SUBDOMAIN", False))
    x_limits = _normalize_limits(cfg.get("X_LIMITS", None))
    y_limits = _normalize_limits(cfg.get("Y_LIMITS", None))
    z_limits = _normalize_limits(cfg.get("Z_LIMITS", None))

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

    processed_nek_base_dir = project_root / Path(cfg.get("PROCESSED_NEK_BASE_DIR", "data/processed_nek"))

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

        elem_idx = None
        if subdomain:
            elem_idx = _select_elements_in_bounds(ds.msh, x_limits, y_limits, z_limits)
            local_total = int(ds.msh.nelv)
            local_keep = int(elem_idx.size)
            global_total = comm.allreduce(local_total, op=MPI.SUM)
            global_keep = comm.allreduce(local_keep, op=MPI.SUM)
            if rank == 0:
                print(f"[rank0] Cropping elements: kept {global_keep}/{global_total}")
        else:
            local_total = int(ds.msh.nelv)
            global_total = comm.allreduce(local_total, op=MPI.SUM)
            if rank == 0:
                print(f"[rank0] Cropping elements: kept {global_total}/{global_total}")

        # Global bounds (before cropping)
        x_local = np.asarray(ds.msh.x)
        y_local = np.asarray(ds.msh.y)
        z_local = np.asarray(ds.msh.z)
        x_min = comm.allreduce(float(np.min(x_local)), op=MPI.MIN)
        x_max = comm.allreduce(float(np.max(x_local)), op=MPI.MAX)
        y_min = comm.allreduce(float(np.min(y_local)), op=MPI.MIN)
        y_max = comm.allreduce(float(np.max(y_local)), op=MPI.MAX)
        z_min = comm.allreduce(float(np.min(z_local)), op=MPI.MIN)
        z_max = comm.allreduce(float(np.max(z_local)), op=MPI.MAX)
        if rank == 0:
            print(
                f"[rank0] Global bounds: "
                f"x=[{x_min}, {x_max}] "
                f"y=[{y_min}, {y_max}] "
                f"z=[{z_min}, {z_max}]"
            )

        # Build augmented Nek field (cropped by elements if requested)
        derived_keys = []
        base_field_names = {"u", "v", "T", "p"}
        base_field_names.update(ds.scalar_names)
        for name in point_data.keys():
            if name not in base_field_names:
                derived_keys.append(name)

        nek_case = make_case_with_base_dir(case, processed_nek_base_dir)
        nek_dir = folder(nek_case)
        if rank == 0:
            nek_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        out_fld_path = nek_dir / f"{file_name}_aug0.f{ts:05d}"
        write_mesh = ts == int(ts_list[0])

        has_elems = elem_idx is None or elem_idx.size > 0
        write_comm = comm.Split(color=1 if has_elems else MPI.UNDEFINED, key=rank)

        if has_elems:
            x_sub = _subset_elements(ds.msh.x, elem_idx)
            y_sub = _subset_elements(ds.msh.y, elem_idx)
            z_sub = _subset_elements(ds.msh.z, elem_idx)
            msh_sub = Mesh(write_comm, x=x_sub, y=y_sub, z=z_sub)

            fld_sub = FieldRegistry(write_comm)
            fld_sub.fields["vel"] = [_subset_elements(f, elem_idx) for f in ds.fld.fields.get("vel", [])]
            fld_sub.fields["pres"] = [_subset_elements(f, elem_idx) for f in ds.fld.fields.get("pres", [])]
            fld_sub.fields["temp"] = [_subset_elements(f, elem_idx) for f in ds.fld.fields.get("temp", [])]
            fld_sub.fields["scal"] = [_subset_elements(f, elem_idx) for f in ds.fld.fields.get("scal", [])]

            for name in derived_keys:
                fld_sub.fields["scal"].append(_subset_elements(point_data[name], elem_idx))

            fld_sub.t = ds.fld.t
            fld_sub.update_vars()

            pynekwrite(str(out_fld_path), write_comm, msh=msh_sub, fld=fld_sub, istep=ts, write_mesh=write_mesh)

            if write_comm.rank == 0:
                base_names = _base_scalar_names(ds.fld, ds.scalar_names)
                aug_names = base_names + derived_keys
                names_path = out_fld_path.with_suffix(".scalars.txt")
                names_path.write_text("\n".join(aug_names) + "\n", encoding="utf-8")

        if write_comm not in (None, MPI.COMM_NULL):
            write_comm.Free()

        # Convert to flat columns for pandas (rank-local)
        data = {
            "x": _subset_elements(ds.x, elem_idx).ravel(order="C"),
            "y": _subset_elements(ds.y, elem_idx).ravel(order="C"),
            "z": _subset_elements(ds.z, elem_idx).ravel(order="C"),
        }

        n = data["x"].size

        for k, arr in point_data.items():
            v = _subset_elements(arr, elem_idx).ravel(order="C")
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
