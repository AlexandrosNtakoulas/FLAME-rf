from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from mpi4py import MPI

from FLAME.datasets import _cantera_temperature_bounds, _domain_width


def _find_project_root(start: Path) -> Path:
    """Find repo root by walking up until 'applications' directory exists."""
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "applications").is_dir():
            return p
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)

try:
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.utils import extrude_2d_sem_mesh
    from pysemtools.interpolation.probes import Probes
    import pysemtools.interpolation.pointclouds as pcs
    import pysemtools.interpolation.utils as interp_utils
    from pysemtools.io.ppymech.neksuite import pynekread
except ModuleNotFoundError:
    pysemtools_local = PROJECT_ROOT / "pySEMTools"
    if str(pysemtools_local) not in sys.path:
        sys.path.append(str(pysemtools_local))
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.utils import extrude_2d_sem_mesh
    from pysemtools.interpolation.probes import Probes
    import pysemtools.interpolation.pointclouds as pcs
    import pysemtools.interpolation.utils as interp_utils
    from pysemtools.io.ppymech.neksuite import pynekread


def _build_case_paths(cfg: dict, project_root: Path) -> tuple[Path, Path, Path, str, int, list[int]]:
    data_base_dir = project_root / Path(cfg["DATA_BASE_DIR"])
    output_base_dir = project_root / Path(cfg["OUTPUT_BASE_DIR"])

    phi = float(cfg["PHI"])
    lat_size = str(cfg["LAT_SIZE"])
    post = bool(cfg["POST"])

    file_prefix = cfg.get("FILE_PREFIX")
    if not file_prefix:
        file_prefix = "po_postPremix0" if post else "premix0"
    file_index_pad = int(cfg.get("FILE_INDEX_PAD", 5))
    geom_time_step = int(cfg.get("GEOM_TIME_STEP", 1))

    phi_dir = f"phi{phi:.2f}"
    lat_tag = f"h400x{lat_size}_ref"
    case_dir = data_base_dir / phi_dir / lat_tag

    phi_tag = f"phi_{phi:.2f}"
    out_dir = output_base_dir / phi_tag / lat_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_file = case_dir / f"{file_prefix}.f{str(geom_time_step).zfill(file_index_pad)}"

    multi = bool(cfg["MULTI_TIME_STEP"])
    if multi:
        time_steps = list(range(int(cfg["TIME_STEP_START"]), int(cfg["TIME_STEP_END"]) + 1))
    else:
        time_steps = [int(cfg["TIME_STEP"])]

    return case_dir, out_dir, mesh_file, file_prefix, file_index_pad, time_steps


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    project_root = PROJECT_ROOT
    os.chdir(str(project_root))

    config_path = project_root / "applications" / "preprocessing" / "nek2structured" / "nek2structured.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    case_dir, out_dir, mesh_file, file_prefix, file_index_pad, requested_steps = _build_case_paths(cfg, project_root)

    if rank == 0:
        print(f"Input case dir: {case_dir}")
        print(f"Output dir: {out_dir}")
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh/geometry file not found: {mesh_file}")
        print(f"Mesh file: {mesh_file}")

    x_bbox = list(cfg["X_BBOX"])
    y_bbox = list(cfg["Y_BBOX"])
    z_bbox = list(cfg["Z_BBOX"])
    nx = int(cfg["NX"])
    ny = int(cfg["NY"])
    nz = int(cfg["NZ"])
    grid_mode = str(cfg.get("GRID_MODE", "equal"))
    grid_gain = float(cfg.get("GRID_GAIN", 1.0))

    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode=grid_mode, gain=grid_gain)
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode=grid_mode, gain=grid_gain)
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode=grid_mode, gain=grid_gain)
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

    points_path = out_dir / str(cfg.get("POINTS_FNAME", "points.hdf5"))
    if rank == 0:
        with h5py.File(points_path, "w") as f:
            f.attrs["nx"] = nx
            f.attrs["ny"] = ny
            f.attrs["nz"] = nz
            f.create_dataset("x", data=x)
            f.create_dataset("y", data=y)
            f.create_dataset("z", data=z)
        print(f"Wrote points file: {points_path}")
    comm.Barrier()

    xyz = interp_utils.transform_from_array_to_list(nx, ny, nz, [x, y, z]) if rank == 0 else None

    file_pairs: list[tuple[int, Path]] = []
    for ts in requested_steps:
        p = case_dir / f"{file_prefix}.f{str(ts).zfill(file_index_pad)}"
        if p.exists():
            file_pairs.append((ts, p))

    if rank == 0:
        missing = [ts for ts in requested_steps if ts not in {t for t, _ in file_pairs}]
        if missing:
            print(f"Warning: {len(missing)} files missing; they will be skipped.")
            for ts in missing[:5]:
                print(f"  missing timestep: {ts}")
        print(f"Interpolating {len(file_pairs)} files.")

    if not file_pairs:
        raise FileNotFoundError("No Nek files found for interpolation.")

    msh = Mesh(comm, create_connectivity=False)
    pynekread(str(mesh_file), comm, msh=msh)

    extrude_2d = bool(cfg.get("EXTRUDE_2D", True))
    mesh_is_2d = msh.x.shape[1] == 1
    if extrude_2d and mesh_is_2d:
        lz = cfg.get("EXTRUDE_LZ", None)
        lz = int(lz) if lz is not None else int(msh.lx)
        msh_use = extrude_2d_sem_mesh(comm, lz=lz, msh=msh)
        if rank == 0:
            print(f"Extruded 2D mesh to lz={lz} for interpolation.")
    else:
        msh_use = msh
        lz = None
        if rank == 0 and mesh_is_2d:
            print("Mesh is 2D (lz=1). Set EXTRUDE_2D: true to avoid interpolation issues.")

    output_file = out_dir / f"{str(cfg.get('OUTPUT_FILE_STEM', 'structured_fields'))}.hdf5"
    probes = Probes(
        comm,
        output_fname=str(output_file),
        probes=xyz,
        msh=msh_use,
        write_coords=bool(cfg.get("WRITE_COORDS", True)),
        point_interpolator_type=str(cfg.get("POINT_INTERPOLATOR_TYPE", "multiple_point_legendre_numpy")),
        max_pts=int(cfg.get("MAX_PTS", 256)),
        find_points_iterative=list(cfg.get("FIND_POINTS_ITERATIVE", [False, 5000])),
        find_points_comm_pattern=str(cfg.get("FIND_POINTS_COMM_PATTERN", "point_to_point")),
        elem_percent_expansion=float(cfg.get("ELEM_PERCENT_EXPANSION", 0.01)),
        global_tree_type=str(cfg.get("GLOBAL_TREE_TYPE", "rank_bbox")),
        global_tree_nbins=int(cfg.get("GLOBAL_TREE_NBINS", 1024)),
        use_autograd=bool(cfg.get("USE_AUTOGRAD", False)),
        find_points_tol=float(cfg.get("FIND_POINTS_TOL", 1.0e-12)),
        find_points_max_iter=int(cfg.get("FIND_POINTS_MAX_ITER", 50)),
        local_data_structure=str(cfg.get("LOCAL_DATA_STRUCTURE", "kdtree")),
        use_oriented_bbox=bool(cfg.get("USE_ORIENTED_BBOX", False)),
    )

    fields_to_interpolate = cfg.get("FIELDS_TO_INTERPOLATE", ["all"])
    if fields_to_interpolate != ["all"]:
        fields_to_interpolate = [str(f) for f in fields_to_interpolate]

    add_progress_var = bool(cfg.get("ADD_PROGRESS_VAR", True))
    temp_field_name = str(cfg.get("TEMP_FIELD_NAME", "t"))
    progress_field_name = str(cfg.get("PROGRESS_FIELD_NAME", "progress_var"))

    t_cold_nd = t_hot_nd = None
    if add_progress_var:
        t_cold_cfg = cfg.get("PROGRESS_T_COLD_ND", None)
        t_hot_cfg = cfg.get("PROGRESS_T_HOT_ND", None)
        if t_cold_cfg is not None and t_hot_cfg is not None:
            t_cold_nd = float(t_cold_cfg)
            t_hot_nd = float(t_hot_cfg)
        elif rank == 0:
            phi = float(cfg["PHI"])
            width = _domain_width(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            t_cold_nd, t_hot_nd, _, _ = _cantera_temperature_bounds(
                cantera_file=str(project_root / Path(cfg.get("CANTERA_YAML", "FLAME/chemical_mech/BurkeH2.yaml"))),
                phi=phi,
                t_ref=float(cfg.get("T_REF", 300.0)),
                p_ref=float(cfg.get("P_REF", 1.0e5)),
                width=width,
                fuel=str(cfg.get("PROGRESS_FUEL", "H2")),
                oxidizer=str(cfg.get("PROGRESS_OXIDIZER", "O2:0.21, N2:0.79")),
                loglevel=int(cfg.get("PROGRESS_LOGLEVEL", 0)),
            )

        t_cold_nd = comm.bcast(t_cold_nd, root=0)
        t_hot_nd = comm.bcast(t_hot_nd, root=0)
        if rank == 0:
            print(
                f"Progress-var bounds: Tcold_nd={t_cold_nd:.6g}, Thot_nd={t_hot_nd:.6g} "
                f"(field='{temp_field_name}' -> '{progress_field_name}')"
            )

    output_file_stem = str(cfg.get("OUTPUT_FILE_STEM", "structured_fields"))
    for ts, fname in file_pairs:
        probes.written_file_counter = int(ts) - 1
        if rank == 0:
            print(f"=== Interpolating {fname.name} ===")

        fld = FieldRegistry(comm)
        pynekread(str(fname), comm, msh=msh, fld=fld)

        fld_use = fld
        if extrude_2d and mesh_is_2d:
            fld_use = extrude_2d_sem_mesh(comm, lz=lz, fld=fld)
            fld_use.t = fld.t

        if fields_to_interpolate == ["all"]:
            field_names = list(fld_use.registry.keys())
        else:
            field_names = []
            for name in fields_to_interpolate:
                if name not in fld_use.registry:
                    raise KeyError(
                        f"Field {name!r} not found in registry. Available: {list(fld_use.registry.keys())}"
                    )
                field_names.append(name)

        field_list = [fld_use.registry[name] for name in field_names]
        probes.interpolate_from_field_list(
            fld_use.t,
            field_list,
            comm,
            write_data=True,
            field_names=field_names,
        )

        if add_progress_var:
            comm.Barrier()
            out_h5 = out_dir / f"{output_file_stem}{str(ts).zfill(file_index_pad)}.hdf5"
            if rank == 0:
                if not out_h5.exists():
                    raise FileNotFoundError(f"Expected interpolated output not found: {out_h5}")
                with h5py.File(out_h5, "r+") as f:
                    if temp_field_name not in f:
                        raise KeyError(
                            f"Temperature field '{temp_field_name}' not found in {out_h5.name}. "
                            f"Available: {list(f.keys())}"
                        )
                    t_data = np.asarray(f[temp_field_name][:], dtype=np.float64)
                    denom = float(t_hot_nd - t_cold_nd)
                    if denom == 0.0:
                        raise ValueError("Invalid progress-variable bounds: Thot_nd == Tcold_nd")
                    progress = (t_data - float(t_cold_nd)) / denom
                    if progress_field_name in f:
                        del f[progress_field_name]
                    f.create_dataset(progress_field_name, data=progress)
            comm.Barrier()

    if rank == 0:
        print(f"Done. Processed {len(file_pairs)} timestep files.")


if __name__ == "__main__":
    main()
