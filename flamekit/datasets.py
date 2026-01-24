from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from mpi4py import MPI

import cantera as ct
import pyvista as pv

from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.field import FieldRegistry

if TYPE_CHECKING:
    from flamekit.io_fronts import Case


# ======================================================================================
# Helpers
# ======================================================================================

def _unwrap_scalar(x: Any) -> Any:
    """Unwrap Nek scalar fields that may be stored as [arr] or arr."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


_PROGRESS_TEMP_CACHE: Dict[Tuple[Any, ...], Tuple[float, float, float, float]] = {}


def _domain_width(x: np.ndarray, y: np.ndarray) -> float:
    x_span = float(np.max(x) - np.min(x))
    y_span = float(np.max(y) - np.min(y))
    width = x_span if x_span > 0.0 else y_span
    if width <= 0.0:
        width = 1.0
    return width


def _resolve_mechanism_path(cantera_file: str) -> Path:
    mech_path = Path(cantera_file).expanduser()
    if not mech_path.is_absolute():
        # assume flamekit repo layout: <repo>/flamekit/datasets.py
        project_root = Path(__file__).resolve().parents[1]
        mech_path = (project_root / mech_path).resolve()
    if not mech_path.exists():
        raise FileNotFoundError(f"Cantera mechanism not found: {mech_path}")
    return mech_path


def _cantera_temperature_bounds(
    *,
    cantera_file: str,
    phi: float,
    t_ref: float,
    p_ref: float,
    width: float,
    fuel: str,
    oxidizer: str,
    loglevel: int,
) -> Tuple[float, float, float, float]:
    """
    Compute reference quantities from a 1-D Cantera FreeFlame.

    Returns
    -------
    t_cold_nd, t_hot_nd, s_l_cm_s, l_ref_cm
    where t_*_nd are nondimensional (divided by t_ref).
    """
    mech_path = _resolve_mechanism_path(cantera_file)
    cache_key = (
        str(mech_path),
        float(phi),
        float(t_ref),
        float(p_ref),
        float(width),
        fuel,
        oxidizer,
    )
    cached = _PROGRESS_TEMP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    gas = ct.Solution(str(mech_path))
    gas.TP = t_ref, p_ref
    gas.set_equivalence_ratio(phi, fuel, oxidizer)

    flame = ct.FreeFlame(gas, width=width)
    flame.inlet.T = t_ref
    flame.inlet.X = gas.X
    flame.set_refine_criteria(curve=0.2, slope=0.1, ratio=2)
    flame.solve(loglevel=loglevel, auto=True)

    t_cold = float(flame.T[0])
    t_hot = float(flame.T[-1])
    t_cold_nd = t_cold / t_ref
    t_hot_nd = t_hot / t_ref

    # laminar flame speed
    if hasattr(flame, "velocity"):
        s_l_m_s = float(flame.velocity[0])
    else:
        # older Cantera API
        s_l_m_s = float(flame.u[0])

    # a practical "reference thickness"
    grid = np.asarray(flame.grid, dtype=float)
    dTdx = np.gradient(flame.T, grid)
    max_grad = float(np.max(np.abs(dTdx)))
    if max_grad <= 0.0:
        raise ValueError("Invalid temperature gradient for L_ref computation")
    l_ref_m = (t_hot - t_cold) / max_grad

    s_l_cm_s = s_l_m_s * 100.0
    l_ref_cm = l_ref_m * 100.0

    _PROGRESS_TEMP_CACHE[cache_key] = (t_cold_nd, t_hot_nd, s_l_cm_s, l_ref_cm)
    return t_cold_nd, t_hot_nd, s_l_cm_s, l_ref_cm


# ======================================================================================
# SEMDataset
# ======================================================================================

class SEMDataset:
    """
    Optimized workflow:
      - Build mesh + coef once in __init__
      - Reload only fields per timestep via reload_timestep()
      - Build PyVista grid connectivity once via build_pyvista_grid_2d()
      - Update point_data per timestep and contour without rebuilding connectivity

    Notes on MPI:
      - pySEMTools distributes elements across ranks when created with comm.
      - Therefore all arrays in this class are **rank-local partitions** when running with MPI size>1.
    """

    def __init__(
        self,
        *,
        folder_name: str,
        file_name: str,
        time_step: int,
        comm: Optional[MPI.Comm] = None,
        scalar_names: Optional[List[str]] = None,
        geometry_step: int = 1,
    ) -> None:
        self.comm = comm
        self.rank = comm.rank if comm is not None else 0
        self.size = comm.size if comm is not None else 1

        self.msh = Mesh(comm, create_connectivity=False)
        self.fld = FieldRegistry(comm)

        self.scalar_names: List[str] = list(scalar_names) if scalar_names else []
        self.scalar_idx: Dict[str, int] = {name: i for i, name in enumerate(self.scalar_names)}

        self.file_name = file_name
        self.folder_name = folder_name
        self.time_step = int(time_step)

        # Resolve folder relative to repo root if needed
        project_root = Path(__file__).resolve().parents[1]
        self.folder = (project_root / Path(folder_name).expanduser()).resolve()

        # Mesh (geometry) is read once
        gname = self.folder / f"{self.file_name}0.f{int(geometry_step):05d}"
        pynekread(str(gname), comm, msh=self.msh)

        # Operators depend only on mesh
        self.coef = Coef(self.msh, comm)

        # Coordinates (rank-local partition)
        self.x, self.y, self.z = self.msh.x, self.msh.y, self.msh.z

        # Fields are loaded per timestep
        self.u = self.v = self.p = self.t = None  # set in reload_timestep
        self.scalars = None

        # PyVista grid cache
        self._pv_grid: Optional[pv.UnstructuredGrid] = None
        self._pv_dims: Optional[Tuple[int, int, int, int]] = None  # (nelv,nz,ny,nx)

    # ----------------------------------------------------------------------------------
    # I/O per timestep
    # ----------------------------------------------------------------------------------
    # Utils
    from pysemtools.datatypes.utils import write_fld_subdomain_from_list    
    def reload_timestep(self, time_step: int, subdomain: bool) -> None:
        self.time_step = int(time_step)
        fname = self.folder / f"{self.file_name}0.f{self.time_step:05d}"
        pynekread(str(fname), self.comm, fld=self.fld, overwrite_fld=True)
        
        if subdomain is True:
            # Write the data in a subdomain and with a different order than what was read
            fout = 'subdomains0.f00001'
            self.write_fld_subdomain_from_list(fout, self.comm, self.msh, field_list=[self.fld.registry['u'],self.fld.registry['v'],self.fld.registry['w']], subdomain=[[-1, 1], [-1, 1], [0, 0.5]])
        
        vel = self.fld.fields.get("vel", None)
        if vel is None:
            raise KeyError("Velocity field 'vel' not found in file.")
        self.u, self.v = vel[0], vel[1]

        self.p = _unwrap_scalar(self.fld.fields.get("pres", None))
        self.t = _unwrap_scalar(self.fld.fields.get("temp", None))
        if self.p is None:
            raise KeyError("Pressure field 'pres' not found in file.")
        if self.t is None:
            raise KeyError("Temperature field 'temp' not found in file.")

        self.scalars = self.fld.fields.get("scal", None)

    # ----------------------------------------------------------------------------------
    # SEM derivatives
    # ----------------------------------------------------------------------------------

    def grad2d_sem(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dfdx = self.coef.dudxyz(f, self.coef.drdx, self.coef.dsdx)
        dfdy = self.coef.dudxyz(f, self.coef.drdy, self.coef.dsdy)
        return dfdx, dfdy

    def hess2d_sem(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dfdx, dfdy = self.grad2d_sem(f)
        d2xx = self.coef.dudxyz(dfdx, self.coef.drdx, self.coef.dsdx)
        d2xy = self.coef.dudxyz(dfdx, self.coef.drdy, self.coef.dsdy)
        d2yy = self.coef.dudxyz(dfdy, self.coef.drdy, self.coef.dsdy)
        return d2xx, d2xy, d2yy

    # ----------------------------------------------------------------------------------
    # Scalars
    # ----------------------------------------------------------------------------------

    def get_scalar(self, name: str) -> np.ndarray:
        if self.scalars is None:
            raise ValueError("No scalar fields ('scal') found in the dataset.")
        if name not in self.scalar_idx:
            raise KeyError(
                f"Scalar {name!r} not found in scalar_names. "
                f"Available names (len={len(self.scalar_names)}): {self.scalar_names[:10]}..."
            )
        i = int(self.scalar_idx[name])
        if i >= len(self.scalars):
            raise IndexError(
                f"Scalar index for {name!r} is {i}, but only {len(self.scalars)} scalar fields exist "
                f"in this file (FieldRegistry 'scal'). Check your SCALARS list."
            )
        return np.asarray(self.scalars[i])

    # ----------------------------------------------------------------------------------
    # Progress variable
    # ----------------------------------------------------------------------------------

    def compute_progress_var_array(
        self,
        *,
        cantera_file: str,
        phi: float,
        t_ref: float,
        p_ref: float,
        fuel: str = "H2",
        oxidizer: str = "O2:0.21, N2:0.79",
        loglevel: int = 0,
    ) -> np.ndarray:
        """
        progress_var = (T - Tcold_nd) / (Thot_nd - Tcold_nd)

        Assumes self.t is nondimensional temperature (T/T_ref).
        """
        width = _domain_width(np.asarray(self.x), np.asarray(self.y))
        comm = self.comm
        rank = self.rank

        if rank == 0:
            t_cold_nd, t_hot_nd, _, _ = _cantera_temperature_bounds(
                cantera_file=cantera_file,
                phi=float(phi),
                t_ref=float(t_ref),
                p_ref=float(p_ref),
                width=width,
                fuel=fuel,
                oxidizer=oxidizer,
                loglevel=loglevel,
            )
        else:
            t_cold_nd = t_hot_nd = None

        if comm is not None:
            t_cold_nd = comm.bcast(t_cold_nd, root=0)
            t_hot_nd = comm.bcast(t_hot_nd, root=0)

        denom = float(t_hot_nd - t_cold_nd)
        if abs(denom) < 1e-12:
            raise ValueError("Invalid temperature bounds for progress_var computation")

        T = np.asarray(self.t)  # nondimensional
        return (T - float(t_cold_nd)) / denom

    # ----------------------------------------------------------------------------------
    # PyVista: build connectivity once, update point_data each timestep
    # ----------------------------------------------------------------------------------

    def build_pyvista_grid_2d(self) -> pv.UnstructuredGrid:
        """
        Build and cache a PyVista UnstructuredGrid (QUADs) once.

        Assumes 2D data stored as (nelv, nz=1, ny, nx) on each rank.
        """
        if self._pv_grid is not None:
            return self._pv_grid

        x = np.asarray(self.x)
        y = np.asarray(self.y)
        z = np.asarray(self.z)

        nelv, nz, ny, nx = x.shape
        if nz != 1:
            raise ValueError(f"Expected nz=1 for 2D; got nz={nz}")

        x2 = x[:, 0, :, :]
        y2 = y[:, 0, :, :]
        z2 = z[:, 0, :, :]

        points = np.column_stack([
            x2.ravel(order="C"),
            y2.ravel(order="C"),
            z2.ravel(order="C"),
        ])

        def idx(e: int, j: int, i: int) -> int:
            return ((e * ny + j) * nx + i)

        cells: List[int] = []
        cell_types: List[int] = []

        for e in range(nelv):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    n0 = idx(e, j, i)
                    n1 = idx(e, j, i + 1)
                    n2 = idx(e, j + 1, i + 1)
                    n3 = idx(e, j + 1, i)
                    cells.extend([4, n0, n1, n2, n3])
                    cell_types.append(pv.CellType.QUAD)

        if not cells:
            raise RuntimeError("UnstructuredGrid has no cells – check mesh dimensions.")

        grid = pv.UnstructuredGrid(
            np.asarray(cells, dtype=np.int64),
            np.asarray(cell_types, dtype=np.uint8),
            points,
        )
        self._pv_grid = grid
        self._pv_dims = (nelv, nz, ny, nx)
        return grid

    def update_pyvista_point_data(self, **arrays: np.ndarray) -> None:
        """
        Update cached PyVista grid point_data arrays (rank-local).

        Each array must match SEM point layout and is raveled with order="C".
        """
        grid = self.build_pyvista_grid_2d()
        for name, arr in arrays.items():
            if arr is None:
                continue
            grid.point_data[name] = np.asarray(arr).ravel(order="C")

    def extract_flame_front(
        self,
        *,
        c_level: float,
        scalar_name: str,
        include_point_data: bool = True,
    ) -> pd.DataFrame:
        """
        Contour the cached grid on scalar_name at c_level (rank-local).

        In MPI, many ranks do not intersect the iso-surface. In that case, an empty DataFrame is returned.
        """
        grid = self.build_pyvista_grid_2d()
        if scalar_name not in grid.point_data:
            raise ValueError(
                f"{scalar_name!r} not found in grid point_data. "
                f"Call update_pyvista_point_data({scalar_name}=...) first."
            )

        Tmin, Tmax = grid.get_data_range(scalar_name)
        if not (Tmin <= c_level <= Tmax):
            return pd.DataFrame({"x": [], "y": [], "z": []})

        iso = grid.contour(scalars=scalar_name, isosurfaces=[float(c_level)])
        pts = iso.points

        out: Dict[str, Any] = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
        if include_point_data:
            for name, arr in iso.point_data.items():
                out[name] = np.asarray(arr)
        return pd.DataFrame(out)

    # ==================================================================================
    # Reaction rates (single public SEM API)
    # ==================================================================================

    def _reaction_rates_core_flat(
        self,
        *,
        cantera_file: str,
        T_nd_flat: np.ndarray,                # (N,) nondimensional T=T/T_ref
        Y_full_flat: np.ndarray,              # (N, n_species_mech) mass fractions (full mechanism)
        t_ref: float,
        p_ref: float,
    ) -> Dict[str, np.ndarray]:
        """
        Internal core: Cantera loop on flattened arrays.

        Returns flat arrays:
          omega_<species>          (N,) [kg/m^3/s]
          heat_conductivity        (N,) [W/m/K]
          thermal_diffusivity      (N,) [cm^2/s]
        """
        gas = ct.Solution(cantera_file)
        n_points = int(T_nd_flat.size)
        n_species = gas.n_species
        MW = gas.molecular_weights  # kg/kmol

        omega_mass = np.zeros((n_points, n_species), dtype=float)
        heat_cond = np.zeros(n_points, dtype=float)
        thermal_diff = np.zeros(n_points, dtype=float)

        T_K = T_nd_flat * float(t_ref)
        P = float(p_ref)

        # A small safety normalisation (optional)
        Y = np.asarray(Y_full_flat, dtype=float)
        Y = np.clip(Y, 0.0, None)
        row_sum = np.sum(Y, axis=1)
        # Avoid divide-by-zero; if sum==0 keep as-is (Cantera may error; that's a data issue)
        mask = row_sum > 0.0
        Y[mask] = (Y[mask].T / row_sum[mask]).T

        for i in range(n_points):
            gas.TPY = float(T_K[i]), P, Y[i, :]
            omega_mass[i, :] = gas.net_production_rates * MW  # kmol/m3/s -> kg/m3/s
            heat_cond[i] = gas.thermal_conductivity
            if gas.density > 0 and gas.cp_mass > 0:
                alpha_m2_s = heat_cond[i] / (gas.density * gas.cp_mass)
            else:
                alpha_m2_s = np.nan
            thermal_diff[i] = alpha_m2_s * 1.0e4  # m^2/s -> cm^2/s

        out: Dict[str, np.ndarray] = {}
        for k, sp in enumerate(gas.species_names):
            out[f"omega_{sp}"] = omega_mass[:, k]
        out["heat_conductivity"] = heat_cond
        out["thermal_diffusivity"] = thermal_diff
        return out

    def compute_reaction_rates(
        self,
        *,
        cantera_file: str,
        species_list: List[str],
        t_ref: float,
        p_ref: float,
        phi_loc_from: Tuple[str, str] = ("H2", "O2"),
        F_O_stoich: float = (0.02851163 / 0.2262686),
    ) -> Dict[str, np.ndarray]:
        """
        Compute Cantera net production rates (mass basis) and some transport properties.

        Parameters
        ----------
        species_list:
            Names of scalar fields in the Nek output holding **mass fractions**.
            These are mapped into the full Cantera mechanism vector. Species not listed are set to 0.

        Returns
        -------
        dict of SEM-shaped arrays (rank-local):
          omega_<species> for all mechanism species, heat_conductivity, thermal_diffusivity, and phi_loc.
        """
        gas = ct.Solution(cantera_file)
        mech_index = {name: i for i, name in enumerate(gas.species_names)}

        # Flatten T
        T_nd = np.asarray(self.t)
        T_nd_flat = T_nd.ravel(order="C")
        n_points = int(T_nd_flat.size)

        # Build full Y for mechanism
        Y_full = np.zeros((n_points, gas.n_species), dtype=float)

        for sp in species_list:
            if sp not in mech_index:
                raise KeyError(f"Species {sp!r} not found in Cantera mechanism.")
            y_sp = np.asarray(self.get_scalar(sp)).ravel(order="C")
            if y_sp.size != n_points:
                raise ValueError(f"Mass fraction field {sp!r} has wrong size (got {y_sp.size}, expected {n_points}).")
            Y_full[:, mech_index[sp]] = y_sp

        out_flat = self._reaction_rates_core_flat(
            cantera_file=cantera_file,
            T_nd_flat=T_nd_flat,
            Y_full_flat=Y_full,
            t_ref=float(t_ref),
            p_ref=float(p_ref),
        )

        # phi_loc (optional proxy, based on mass fraction ratio)
        a, b = phi_loc_from
        if a in species_list and b in species_list:
            Ya = np.asarray(self.get_scalar(a)).ravel(order="C")
            Yb = np.asarray(self.get_scalar(b)).ravel(order="C")
            phi_loc = (Ya / np.maximum(Yb, 1e-30)) / float(F_O_stoich)
        else:
            phi_loc = np.full(n_points, np.nan, dtype=float)
        out_flat["phi_loc"] = phi_loc

        # Reshape to SEM partition shape
        out_sem: Dict[str, np.ndarray] = {}
        for k, v in out_flat.items():
            out_sem[k] = np.asarray(v).reshape(T_nd.shape, order="C")
        return out_sem

    # ==================================================================================
    # One shared feature builder (used by both full-field export and isocontours)
    # ==================================================================================

    def build_point_data_dict(
        self,
        *,
        cantera_inputs: Optional[List[Any]] = None,  # [cantera_file, species_list, t_ref, p_ref]
        phi: Optional[float] = None,
        # include as subset of all fields
        scalar_subset: Optional[List[str]] = None,
        # derived
        compute_progress_var: bool = True,
        compute_vel_jacobian: bool = False,
        compute_vel_hessian: bool = False,
        compute_T_grad: bool = False,
        compute_curv_grad: bool = False,
        compute_local_vel_jacobian: bool = False,
        compute_reaction_rates: bool = False,
        # progress
        progress_fuel: str = "H2",
        progress_oxidizer: str = "O2:0.21, N2:0.79",
        progress_loglevel: int = 0,
        # reaction rates
        reaction_species_list: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build a dict of SEM-shaped arrays to attach as PyVista point_data, or to flatten into a DataFrame.

        This is the **single place** controlling feature availability.
        """
        point_data: Dict[str, np.ndarray] = {}

        # Base fields
        
        point_data["u"] = np.asarray(self.u)
        point_data["v"] = np.asarray(self.v)
        point_data["T"] = np.asarray(self.t)
        point_data["p"] = np.asarray(self.p)
        
        #Scalars
        if scalar_subset:
            for name in scalar_subset:
                point_data[name] = np.asarray(self.get_scalar(name))
        else:
            n = min(len(self.scalar_names), len(self.scalars))
            for i in range(n):
                point_data[self.scalar_names[i]] = np.asarray(self.scalars[i])


        # Progress variable
        if compute_progress_var:
            if cantera_inputs is None or len(cantera_inputs) < 4:
                raise ValueError("cantera_inputs must include [cantera_file, species_list, t_ref, p_ref]")
            if phi is None:
                raise ValueError("phi must be provided when compute_progress_var=True")
            cantera_file = str(cantera_inputs[0])
            t_ref = float(cantera_inputs[2])
            p_ref = float(cantera_inputs[3])
            point_data["progress_var"] = self.compute_progress_var_array(
                cantera_file=cantera_file,
                phi=float(phi),
                t_ref=t_ref,
                p_ref=p_ref,
                fuel=progress_fuel,
                oxidizer=progress_oxidizer,
                loglevel=progress_loglevel,
            )

        # Derivatives / Jacobians
        if compute_T_grad:
            dTdx, dTdy = self.grad2d_sem(np.asarray(self.t))
            point_data["dTdx"] = dTdx
            point_data["dTdy"] = dTdy

        if compute_vel_jacobian:
            dudx, dudy = self.grad2d_sem(np.asarray(self.u))
            dvdx, dvdy = self.grad2d_sem(np.asarray(self.v))
            point_data["dudx"] = dudx
            point_data["dudy"] = dudy
            point_data["dvdx"] = dvdx
            point_data["dvdy"] = dvdy

        if compute_vel_hessian:
            d2u_xx, d2u_xy, d2u_yy = self.hess2d_sem(np.asarray(self.u))
            d2v_xx, d2v_xy, d2v_yy = self.hess2d_sem(np.asarray(self.v))
            point_data["d2u_xx"] = d2u_xx
            point_data["d2u_xy"] = d2u_xy
            point_data["d2u_yy"] = d2u_yy
            point_data["d2v_xx"] = d2v_xx
            point_data["d2v_xy"] = d2v_xy
            point_data["d2v_yy"] = d2v_yy

        if compute_curv_grad:
            curv = self.get_scalar("curvature")
            dcdx, dcdy = self.grad2d_sem(np.asarray(curv))
            point_data["curvature"] = np.asarray(curv)
            point_data["dcurvdx"] = dcdx
            point_data["dcurvdy"] = dcdy

        if compute_local_vel_jacobian:
            u_n = self.get_scalar("u_n")
            u_t = self.get_scalar("u_t")
            dun_dx, dun_dy = self.grad2d_sem(np.asarray(u_n))
            dut_dx, dut_dy = self.grad2d_sem(np.asarray(u_t))
            point_data["u_n"] = np.asarray(u_n)
            point_data["u_t"] = np.asarray(u_t)
            point_data["du_ndx"] = dun_dx
            point_data["du_ndy"] = dun_dy
            point_data["du_tdx"] = dut_dx
            point_data["du_tdy"] = dut_dy

        # Reaction rates
        if compute_reaction_rates:
            if cantera_inputs is None or len(cantera_inputs) < 4:
                raise ValueError("cantera_inputs must include [cantera_file, species_list, t_ref, p_ref]")
            cantera_file = str(cantera_inputs[0])
            t_ref = float(cantera_inputs[2])
            p_ref = float(cantera_inputs[3])

            # choose species list for Y
            # Hardcoded
            species_list = ["H2", "O2", "H2O","H","O","OH","HO2","H2O2","N2"]  # Ddefault

            rr = self.compute_reaction_rates(
                cantera_file=cantera_file,
                species_list=species_list,
                t_ref=t_ref,
                p_ref=p_ref,
            )
            point_data.update(rr)

        return point_data
# ======================================================================================