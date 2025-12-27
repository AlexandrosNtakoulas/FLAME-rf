from __future__ import annotations
import numpy as np
import pandas as pd
from mpi4py import MPI
from typing import Optional, List
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field
from pysemtools.io.ppymech.neksuite import pynekread

import cantera as ct

from pySEMTools.pysemtools.datatypes import FieldRegistry

import pyvista as pv

# ----------------------------------------------------
# SEMDataset: load + access Nek data
# ----------------------------------------------------
class SEMDataset:
    """
    Dataset object representing Nek5000 simulation data read via PySEMTools.

    Attributes
    ----------
    msh : Mesh
        Spectral element mesh object (geometry and connectivity).
    fld : Field
        Field container with solution variables (vel, temp, scalars, etc.).
    coef : Coef
        Derivative operator coefficients for computing gradients.
    comm :
        TODO

    x, y, z : np.ndarray
        Coordinate arrays.
    u, v, w : np.ndarray
        Velocity arrays in x,y,z directions respectively
    t : ndarray
        Temperature array
    scalar_names : list
        Names of scalars

    """
    def __init__(
            self,
            folder_name: str,
            file_name: str,
            time_step: int,
            comm: Optional[MPI.Comm] = None,
            scalar_names: List[str] = None
    ) -> None:
        self.comm = comm
        self.msh = Mesh(comm, create_connectivity=False, )
        self.fld = FieldRegistry(comm)
        self.scalar_names = scalar_names
        self.dataframe = None
        self.file_name = file_name
        self.folder_name = folder_name
        self.time_step = time_step


        gname = f"../{self.folder_name}/{self.file_name}0.f00001"
        fname = f"../{self.folder_name}/{self.file_name}0.f{self.time_step:05d}"
        # read coordinates/mesh from gname (geometry file)
        pynekread(gname, comm, msh=self.msh, fld=self.fld, overwrite_fld=True)
        # read actual field data from fname (time snapshot)
        pynekread(fname, comm, msh=self.msh, fld=self.fld, overwrite_fld=True)
        # Build derivative operators -> look more into this
        self.coef = Coef(self.msh, comm)
        # Cache coords -> delete?
        self.x, self.y, self.z = self.msh.x, self.msh.y, self.msh.z
        # Velocity
        vel = self.fld.fields.get("vel", None)
        self.u, self.v = vel[0], vel[1]
        #Pressure
        self.p = _unwrap_scalar(self.fld.fields.get("pres", None))
        # Temperature
        self.t = _unwrap_scalar(self.fld.fields.get("temp", None))
        # Scalars
        self.scalars = self.fld.fields.get("scal", None)
        return

    def grad2d(self, f: np.ndarray):
        """Return jacobian of scalar field"""
        dfdx = self.coef.dudxyz(f, self.coef.drdx, self.coef.dsdx)
        dfdy = self.coef.dudxyz(f, self.coef.drdy, self.coef.dsdy)
        return dfdx.ravel(), dfdy.ravel()

    def hess2d(self, f: np.ndarray):
        """"Return hessian of scalar field"""
        dfdx, dfdy = self.grad2d(f)
        d2xx = self.coef.dudxyz(dfdx, self.coef.drdx, self.coef.dsdx)
        d2xy = self.coef.dudxyz(dfdx, self.coef.drdy, self.coef.dsdy)
        d2yy = self.coef.dudxyz(dfdy, self.coef.drdy, self.coef.dsdy)
        return d2xx, d2xy, d2yy

    def create_dataframe(
            self,
            compute_vel_jacobian: bool = False,
            compute_vel_hessian: bool = False,
            compute_reaction_rates: bool = False,
            compute_T_grad: bool = False,
            compute_curv_grad: bool = False,
            compute_local_vel_jacobian: bool = False,
            cantera_inputs: Optional[List[str, float]] = None
    ) -> pd.DataFrame:
        """
        Create and return a dataframe with the DNS data

        Parameters
        ----------
        compute_vel_jacobian: boolean

        compute_vel_hessian: boolean

        compute_reaction_rates: boolean

        cantera_inputs: list
            List of cantera file, species list, equivalence ratio, T_ref, p_ref
        """
        # Add basic fields
        data = {
            "x": self.x.reshape(-1),
            "y": self.y.reshape(-1),
            "u": self.u.reshape(-1),
            "v": self.v.reshape(-1),
            "T": self.t.reshape(-1),
            "p": self.p.reshape(-1),
        }
        # Add scalar values
        for i in range(len(self.scalars)):
            data.update(
                {
                    self.scalar_names[i]: np.array(self.scalars[i]).reshape(-1)
                }
            )
        self.dataframe = pd.DataFrame(data)

        if compute_vel_jacobian:
            self.add_vel_jacobian_to_dataframe()
        if compute_vel_hessian:
            self.add_vel_hessian_to_dataframe()
        if compute_reaction_rates:
            self.add_reaction_rates_to_dataframe(
                cantera_file= cantera_inputs[0],
                species_list= cantera_inputs[1],
                t_ref= cantera_inputs[2],
                p_ref= cantera_inputs[3]
            )
        if compute_T_grad:
            self.dataframe["dTdx"], self.dataframe["dTdy"] = self.grad2d(self.t)
        if compute_curv_grad:
            self.dataframe["dcurvdx"], self.dataframe["dcurvdy"] = self.grad2d(self.scalars[10])
        if compute_local_vel_jacobian:
            self.dataframe["du_ndx"], self.dataframe["du_ndy"] = self.grad2d(self.scalars[14])
            self.dataframe["du_tdx"], self.dataframe["du_tdy"] = self.grad2d(self.scalars[15])
        return self.dataframe

    def add_vel_jacobian_to_dataframe(self):
        """Add jacobian of velocity to the dataframe"""
        self.dataframe["dudx"], self.dataframe["dudy"] = self.grad2d(self.u)
        self.dataframe["dvdx"], self.dataframe["dvdy"] = self.grad2d(self.v)

    def add_vel_hessian_to_dataframe(self):
        """Add hessian of velocity to the dataframe"""
        self.dataframe["d2u_xx"], self.dataframe["d2u_xy"], self.dataframe["d2u_yy"] \
            = self.hess2d(self.u)
        self.dataframe["d2v_xx"], self.dataframe["d2v_xy"], self.dataframe["d2v_yy"] \
            = self.hess2d(self.v)

    def add_reaction_rates_to_dataframe(
            self,
            cantera_file: str = None,
            species_list: List[str] = None,
            t_ref: float = None,
            p_ref: int = None
    ):
        """Add reaction rates and lewis number of deficient reactant to the dataframe"""
        if species_list is None:
            species_list = ['H2', 'O2', 'H2O', 'H', 'O', 'OH', 'HO2', 'H2O2', 'N2']
        y_matrix = np.stack([self.dataframe[name].values for name in species_list], axis=1)
        gas = ct.Solution(cantera_file)
        n_points = y_matrix.shape[0]
        n_species = gas.n_species
        reaction_rates_molar = np.zeros((n_points, n_species))  # kmol/m3/s
        reaction_rates_mass = np.zeros((n_points, n_species))   # kg/m3/s
        MW = gas.molecular_weights  # kg/kmol, length = n_species
        for i in range(n_points):
            gas.TPY = self.dataframe["T"].values[i] * t_ref, p_ref, y_matrix[i, :]
            reaction_rates_molar[i, :] = gas.net_production_rates
            # convert molar -> mass [kg/m3/s]
            reaction_rates_mass[i, :] = reaction_rates_molar[i, :] * MW
        F_O_stoich = (0.02851163 / 0.2262686)
        self.dataframe["phi_loc"] = (self.dataframe["H2"] / self.dataframe["O2"]) / F_O_stoich
        for k, sp in enumerate(gas.species_names):
            self.dataframe[f"omega_{sp}"] = reaction_rates_mass[:, k]
        return

    def extract_flame_front(
            self,
            c_level: float = None,
    ) -> pd.DataFrame:
        """
        2D version: build a VTK unstructured QUAD grid from SEM coordinates
        and extract a temperature isocontour at T = c_level.

        Assumes 2D data (one layer in z); z may be constant.
        """

        # ------------------------------------------------------------------
        # 1) Normalize coordinates to shape (nelv, ny, nx)
        # ------------------------------------------------------------------
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        z = np.asarray(self.z)

        # Expect (nelv, nz, ny, nx) with nz = 1 for 2D
        nelv, nz, ny, nx = x.shape
        x = x[:, 0, :, :]
        y = y[:, 0, :, :]
        z = z[:, 0, :, :]


        # ------------------------------------------------------------------
        # 2) Build point array: shape (N_points, 3)
        #    Flatten with loop order (e, j, i) using ravel(order="C")
        # ------------------------------------------------------------------
        points = np.column_stack([
            x.ravel(order="C"),
            y.ravel(order="C"),
            z.ravel(order="C"),
        ])  # (N_points, 3)
        n_points = points.shape[0]
        def idx(e: int, j: int, i: int) -> int:
            """Global point index for (e, j, i) matching ravel(order='C')."""
            return ((e * ny + j) * nx + i)

        # ------------------------------------------------------------------
        # 3) Build QUAD cell connectivity
        # ------------------------------------------------------------------
        cells = []
        cell_types = []

        for e in range(nelv):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    n0 = idx(e, j, i)
                    n1 = idx(e, j, i + 1)
                    n2 = idx(e, j + 1, i + 1)
                    n3 = idx(e, j + 1, i)

                    # VTK legacy format: [n_points_in_cell, p0, p1, p2, p3]
                    cells.extend([4, n0, n1, n2, n3])
                    cell_types.append(pv.CellType.QUAD)

        cells = np.asarray(cells, dtype=np.int64)
        cell_types = np.asarray(cell_types, dtype=np.uint8)

        if cells.size == 0:
            raise RuntimeError("UnstructuredGrid has no cells – check mesh dimensions.")

        # ------------------------------------------------------------------
        # 4) Create UnstructuredGrid
        # ------------------------------------------------------------------
        grid = pv.UnstructuredGrid(cells, cell_types, points)

        # ------------------------------------------------------------------
        # 5) Attach fields as point data
        # ------------------------------------------------------------------
        u = np.asarray(self.u)
        u = u[:,0,:,:]

        v = np.asarray(self.v)
        v = v[:,0,:,:]

        T = np.asarray(self.fld.fields["temp"])
        T = T[0, :, 0, :, :]

        grid.point_data["u"] = u.ravel(order="C")
        grid.point_data["v"] = v.ravel(order="C")
        grid.point_data["T"] = T.ravel(order="C")

        for name, values in zip(self.scalar_names, self.scalars):
            arr = np.asarray(values)
            arr = arr[:, 0, :, :]
            grid.point_data[name] = arr.ravel(order="C")

        if self.dataframe is not None:
            # columns already flattened in the same order as x.reshape(-1)
            skip_cols = set(["x", "y", "u", "v", "T"])
            if self.scalar_names is not None:
                skip_cols.update(self.scalar_names)

            for col in self.dataframe.columns:
                if col in skip_cols:
                    continue  # these are already on the grid

                arr_flat = self.dataframe[col].to_numpy()
                if arr_flat.size != n_points:
                    continue

                # Attach as point-data scalar (or vector if needed later)
                grid.point_data[col] = arr_flat
        # ------------------------------------------------------------------
        # 6) Extract isocontour of the temperature field
        # ------------------------------------------------------------------
        if c_level is None:
            raise ValueError("c_level must be specified for isocontour extraction")

        Tmin, Tmax = grid.get_data_range("T")
        if not (Tmin <= c_level <= Tmax):
            raise ValueError(f"c_level={c_level} is outside T range [{Tmin}, {Tmax}]")

        iso = grid.contour(scalars="T", isosurfaces=[c_level])

        # ------------------------------------------------------------------
        # 7) Build DataFrame with coordinates + all variables on the contour
        # ------------------------------------------------------------------
        pts = iso.points  # shape (N, 3)
        data = {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
        }

        for name, arr in iso.point_data.items():
            data[name] = np.asarray(arr)

        front = pd.DataFrame(data)
        return front

def _unwrap_scalar(x):
    return x[0] if isinstance(x, list) else x