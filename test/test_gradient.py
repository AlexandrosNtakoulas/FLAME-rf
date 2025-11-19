import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from mpi4py import MPI
comm = MPI.COMM_WORLD

import numpy as np
from pysemtools.io.ppymech.neksuite import preadnek
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef


def analytic_function(x, y, z):
    """Analytic test field on [0,1]^3."""
    f = np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(z)
    dfdx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y) * np.exp(z)
    dfdy = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(z)
    dfdz = np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(z)
    return f, dfdx, dfdy, dfdz


def test_gradient_accuracy():
    """Check pySEMTools derivative accuracy with manufactured solution."""
    fname = "../data/v2/premix0.f00001"
    data = preadnek(fname, comm)
    msh = Mesh(comm, data=data)
    del data

    coef = Coef(msh, comm)
    x, y, z = msh.x, msh.y, msh.z
    ndim = msh.gdim

    # Coordinate ranges
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    z0, z1 = z.min(), z.max()
    print(f"x:[{x0:.3e},{x1:.3e}], y:[{y0:.3e},{y1:.3e}], z:[{z0:.3e},{z1:.3e}]")

    # Normalize coordinates to [0,1]
    def safe_norm(a, a0, a1):
        return np.zeros_like(a) if a1 == a0 else (a - a0) / (a1 - a0)

    x_n = safe_norm(x, x0, x1)
    y_n = safe_norm(y, y0, y1)
    z_n = safe_norm(z, z0, z1)

    f, dfdx_n, dfdy_n, dfdz_n = analytic_function(x_n, y_n, z_n)

    # Rescale analytic derivatives back to physical units safely
    def safe_scale(arr, a0, a1):
        return np.zeros_like(arr) if a1 == a0 else arr / (a1 - a0)

    dfdx_true = safe_scale(dfdx_n, x0, x1)
    dfdy_true = safe_scale(dfdy_n, y0, y1)
    dfdz_true = safe_scale(dfdz_n, z0, z1)

    # Compute derivatives with pySEMTools
    if ndim == 3:
        dfdx = coef.dudxyz(f, coef.drdx, coef.dsdx, coef.dtdx)
        dfdy = coef.dudxyz(f, coef.drdy, coef.dsdy, coef.dtdy)
        dfdz = coef.dudxyz(f, coef.drdz, coef.dsdz, coef.dtdz)
    elif ndim == 2:
        dfdx = coef.dudxyz(f, coef.drdx, coef.dsdx)
        dfdy = coef.dudxyz(f, coef.drdy, coef.dsdy)
        dfdz = np.zeros_like(f)
    else:
        raise ValueError(f"Unexpected ndim={ndim}")

    def rel_L2_error(num, ref):
        num = np.nan_to_num(num)
        ref = np.nan_to_num(ref)
        denom = np.sum(ref ** 2)
        if denom == 0:
            return 0.0
        return np.sqrt(np.sum((num - ref) ** 2) / denom)

    err_x = rel_L2_error(dfdx, dfdx_true)
    err_y = rel_L2_error(dfdy, dfdy_true)
    err_z = rel_L2_error(dfdz, dfdz_true)

    print(f"[Derivative Test] L2 Errors: dx={err_x:.2e}, dy={err_y:.2e}, dz={err_z:.2e}")

    tol = 1e-6
    if ndim == 2:
        assert err_x < tol and err_y < tol, \
            f"Derivative test failed (2D): dx={err_x:.2e}, dy={err_y:.2e}"
    else:
        assert err_x < tol and err_y < tol and err_z < tol, \
            f"Derivative test failed (3D): dx={err_x:.2e}, dy={err_y:.2e}, dz={err_z:.2e}"


if __name__ == "__main__":
    test_gradient_accuracy()
