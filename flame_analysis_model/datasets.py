from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.backend import backend
from mpi4py import MPI
from typing import Optional, List, Tuple
import torch
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field
from pysemtools.io.ppymech.neksuite import pynekread

import cantera as ct

from pySEMTools.pysemtools.datatypes import FieldRegistry
from .utils import _unwrap_scalar

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


        gname = f"./{self.folder_name}/{self.file_name}0.f00001"
        fname = f"./{self.folder_name}/{self.file_name}0.f{self.time_step:05d}"
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
        self.w = vel[2] if len(vel) > 2 else np.zeros_like(self.u)
        # Temperature
        self.t = _unwrap_scalar(self.fld.fields.get("temp", None))
        # Scalars
        self.scalars = self.fld.fields.get("scal", None)
        assert (len(self.scalar_names) != (3 + len(self.scalars))), \
            (f"There is a mismatch in the number of variables.  "
             f"You gave {len(self.scalar_names)} "
             f"but the dataset has {len(self.scalars)} scalars")
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
            cantera_inputs: Optional[List[str, float]] = None
    ) -> pd.DataFrame:
        """Create and return a dataframe with the DNS data"""
        # Add basic fields
        data = {
            "x": self.x.reshape(-1),
            "y": self.y.reshape(-1),
            "u": self.u.reshape(-1),
            "v": self.v.reshape(-1),
            "T": self.t.reshape(-1),
        }

        # Add scalar values
        for i in range(len(self.scalar_names)):
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
                eq_ratio= cantera_inputs[2],
                t_ref= cantera_inputs[3],
                p_ref= cantera_inputs[4]
            )
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
            eq_ratio: float = None,
            t_ref: float = 300,
            p_ref: int = 1e05
    ):
        """Add reaction rates and lewis number of deficient reactant to the dataframe"""
        if species_list is None:
            species_list = ['H2', 'O2', 'H2O', 'H', 'O', 'OH', 'HO2', 'H2O2', 'N2']

        chem = ChemistryFeatures(
            mechanism= cantera_file,
            phi= eq_ratio,
            t_ref= t_ref,
            p_ref= p_ref
        )
        y_matrix = np.stack([self.dataframe[name].values for name in species_list], axis=1)
        omegas, def_lewis_num = chem.compute_features(self.dataframe["T"].values, y_matrix)
        for k, sp in enumerate(chem.gas.species_names):
            self.dataframe[f"omega_{sp}"] = omegas[:, k]
        self.dataframe["Le_def"] = def_lewis_num
        return

    @staticmethod
    def band_mask(c: np.ndarray, c_level: float = 0.38, tol: float = 0.01):
        """Return a band mask with a specified level and tolerance"""
        return (c > (c_level - tol)) & (c < (c_level + tol))

    @staticmethod
    def progress_variable_T(t: np.ndarray = None, t_u: float = None, t_b: float = None):
        """Return the temperature-based progress variable"""
        t_u = np.percentile(t, q=2)
        t_b = np.percentile(t, q=98)
        den = t_b - t_u if t_b > t_u else 1.0
        return np.clip((t - t_u) / den, a_min=0.0, a_max=1.0)

    def heat_release_mask(
            self,
            threshold_factor: float = 0.15,
            heat_release_field=None
    ):
        """Return mask where heat release > threshold_factor * max_heat_release"""
        max_hrr = np.max(heat_release_field)
        return heat_release_field > (threshold_factor * max_hrr)

    def extract_flame_front_new(
            self,
            sample_mode: str = "isocontour",
            c_level: float = None,
            tol: float = None,
            hrr_factor: float = None
    ) -> pd.DataFrame:
        """
        Extract an isocontour of the temperature field on the Nek/pySEMTools mesh
        and return all variables on that isocontour as a DataFrame.
        """

        assert self.dataframe is not None, "Dataframe has not been created yet"
        assert sample_mode == "isocontour", "Only 'isocontour' implemented here"
        assert c_level is not None, "For 'isocontour', c_level must be the temperature isovalue"

        grid = pv.StructuredGrid(self.x.reshape(-1), self.y.reshape(-1), self.z.reshape(-1))
        grid.point_data["temp"] = np.asarray(self.fld.fields["temp"]).ravel()
        for i in range(len(self.scalar_names)):
            grid.point_data[self.scalar_names[i]] = np.asarray(self.scalars[i]).ravel()

        # 3) Extract isocontour of the temperature field
        #    Adjust 'temp' if your temperature field has a different name.
        iso = grid.contour(scalars="temp", isosurfaces=[c_level])

        # 4) Build a DataFrame with coordinates + all variables on the contour
        pts = iso.points  # shape (N, 3)
        data = {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
        }

        # Add all point-data arrays present on the contour (temp, vel, scalars,…)
        for name, arr in iso.point_data.items():
            data[name] = np.asarray(arr)

        front = pd.DataFrame(data)
        return front

    def extract_flame_front_dataframe(
            self,
            sample_mode: str = "isocontour",
            c_level: float = None,
            tol: float = None,
            hrr_factor: float = None
    ) -> pd.DataFrame:
        """
        Returns a dataframe of only the flame front.

        sample_mode:
            "isocontour" : use PyVista contour on T at T = c_level
            "progress"   : use band in progress variable
            "hrr"        : use heat-release threshold
        """
        assert (self.dataframe is not None), "Dataframe has not been created yet"

        if sample_mode == "isocontour":
            assert c_level is not None, \
                "For sample_mode='isocontour', c_level must be the temperature isovalue"

            # Read the Nek5000 file via PyVista
            reader = pv.get_reader(f"./{self.folder_name}/{self.file_name}.nek5000")
            reader.set_active_time_value(self.time_step)
            ds = reader.read()  # this is a PyVista dataset (likely StructuredGrid)

            # Extract the isocontour Temperature = c_level
            iso = ds.contour(isosurfaces=[c_level], scalars="Temperature")

            pts = iso.points
            df_dict = {
                "x": pts[:, 0],
                "y": pts[:, 1],
            }

            # Add all point-data variables present on the contour
            # (Temperature, velocity components, species, etc.)
            for name, arr in iso.point_data.items():
                df_dict[name] = np.asarray(arr)

            front = pd.DataFrame(df_dict)
            return front

        # --- existing modes preserved ---
        if sample_mode == "progress":
            c = self.progress_variable_T(self.t)
            mask = self.band_mask(c, c_level, tol).ravel()
            return self.dataframe[mask]

        elif sample_mode == "hrr":
            raise NotImplementedError(
                "sample_mode='hrr' not wired to a stored heat_release_field yet."
            )

        else:
            raise ValueError("Invalid sampling mode: 'isocontour', 'progress', or 'hrr'")




class ChemistryFeatures:
    """"
    Chemistry object used for interfacing with cantera
    TODO
    Attributes
    ----------
    gas: something
        Cantera solution thingy
    phi: float
        Equivalence ration -> TODO This has to match with the ones on the cantera files
    t_ref: float
        Reference temperature
    p_ref: int
        Reference pressure -> TODO Check units
    """
    def __init__(
            self,
            mechanism: str = "chem.yaml",
            phi: float = 1.0,
            t_ref: float = 300.0,
            p_ref: float = 1.0
    ):
        self.gas = ct.Solution(mechanism)
        self.gas.transport_model = "Mix"
        self.phi = phi
        self.t_ref = t_ref
        self.p_ref = p_ref * ct.one_atm

    def compute_features(
            self,
            t: np.ndarray = None,
            y_matrix: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reaction rates and lewis number of deficient reactant"""
        gas = self.gas
        reaction_rates = []
        def_lewis_num = []

        for i in range(y_matrix.shape[0]):
            gas.TPY = t[i] * self.t_ref, self.p_ref, y_matrix[i, :]
            reaction_rates.append(gas.net_production_rates)
            # deficient reactant index
            if self.phi < 1.0:
                i_def = gas.species_index("H2")
            else:
                i_def = gas.species_index("O2")

            alpha = gas.thermal_conductivity / (gas.density * gas.cp)
            def_lewis_num.append(alpha / gas.mix_diff_coeffs[i_def])

        return np.array(reaction_rates), np.array(def_lewis_num)

# ----------------------------------------------------
# Plot2D: quick visualization helpers
# ----------------------------------------------------
class Plot2D:

    @staticmethod
    def plot_field(
            msh,
            field,
            mode="sem",
            plot_name=None,
            levels=100,
            cmap="RdBu_r",
            vmin=None,
            vmax=None
    ):
        """
        TODO Maybe turn this into methods of the SEMDataset class or simply utility functions
        Plot a scalar field on a pySEMTools mesh.

        Parameters
        ----------
        msh : Mesh
            pySEMTools Mesh object (provides x,y arrays).
        field : ndarray or list
            Field data (e.g. fld.registry["u"]).
        mode : {"sem","tri"}
            sem = element-wise pcolormesh (preserve SEM structure).
            tri = tricontourf on flattened points.
        """

        def _to_numpy(a):
            if isinstance(a, list):
                a = a[0]
            if hasattr(a, "detach") and hasattr(a, "cpu"):
                a = a.detach().cpu().numpy()
            return np.asarray(a)

        F = _to_numpy(field)
        X = _to_numpy(msh.x)
        Y = _to_numpy(msh.y)

        if vmin is None: vmin = float(np.nanmin(F))
        if vmax is None: vmax = float(np.nanmax(F))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=160)

        if mode == "sem":
            for e in range(X.shape[0]):
                pc = ax.pcolormesh(
                    X[e, 0, :, :], Y[e, 0, :, :], F[e, 0, :, :],
                    shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
                )
            cbar = fig.colorbar(pc, ax=ax)
        else:
            x, y, f = X.ravel(), Y.ravel(), F.ravel()
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
            tc = ax.tricontourf(x[m], y[m], f[m], levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(tc, ax=ax)

        cbar.set_label("field value")
        ax.set_xlabel("x");
        ax.set_ylabel("y")
        ax.set_title(f"Field plot of {plot_name}({'SEM patches' if mode == 'sem' else 'triangulated'})")
        plt.show()

    @staticmethod
    def print_pearson(df: pd.DataFrame, cols: Optional[List[str]] = None):
        num = df[cols] if cols is not None else df.select_dtypes(include=[np.number])
        print(num.corr(method="pearson").round(3))

    @staticmethod
    def heat_map(df: pd.DataFrame, cols: list[str], max_points: int = 8000):
        from itertools import product
        """
        Pairwise scatter/histogram matrix with optional coloring by progress variable c.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to visualize.
        cols : list of str
            List of column names to plot against each other.
        max_points : int
            Limit number of points (random sample) to keep plots responsive.
        """
        # downsample if necessary
        if len(df) > max_points:
            dfp = df.sample(max_points, random_state=0)
        else:
            dfp = df

        k = len(cols)
        fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k), dpi=140)

        cvals = dfp["c"].values if "c" in dfp.columns else None
        sc = None  # for colorbar handle

        for i, j in product(range(k), range(k)):
            ax = axes[i, j]
            xi, yj = cols[j], cols[i]
            if i == j:
                ax.hist(dfp[xi].values, bins=40, alpha=0.85)
                ax.set_ylabel("")
            else:
                if cvals is None:
                    ax.scatter(dfp[xi].values, dfp[yj].values, s=6, alpha=0.4)
                else:
                    sc = ax.scatter(dfp[xi].values, dfp[yj].values, s=6, alpha=0.5,
                                    c=cvals, cmap="viridis")

            # tidy ticks
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(xi)
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(yj)

        # single colorbar if c is present
        if sc is not None:
            cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
            cb = plt.colorbar(sc, cax=cax)
            cb.set_label("progress variable c")

        plt.show()
