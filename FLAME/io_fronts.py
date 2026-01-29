from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

FRONT_HDF_KEY = "front"

@dataclass(frozen=True)
class Case:
    base_dir: Path
    phi: float
    lat_size: str
    time_step: int
    post: bool = True
    multiple_runs: bool = False
    n_run: int = 0
    h_tag: str = "h400"

def folder(c: Case) -> Path:
    p = c.base_dir / f"phi{c.phi:.2f}" / f"{c.h_tag}x{c.lat_size}_ref"
    if c.multiple_runs:
        p = p / f"RUN0{c.n_run}"
    return p

def front_filename(c: Case, iso: float) -> str:
    tag = "post_" if c.post else ""
    return f"extracted_flame_front_{tag}{c.time_step}_iso_{iso}.hdf5"

def front_path(c: Case, iso: float) -> Path:
    return folder(c) / front_filename(c, iso)

def load_front_csv(
    c: Case,
    iso: float,
    *,
    dtype: str | None = "float32",
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    fp = front_path(c, iso)
    if fp.exists():
        df = pd.read_hdf(fp, key=FRONT_HDF_KEY)
    else:
        csv_fp = fp.with_suffix(".csv")
        if csv_fp.exists():
            df = pd.read_csv(csv_fp)
        else:
            raise FileNotFoundError(f"Missing front file: {fp}")

    if dtype is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].astype(dtype)

    if required_cols:
        missing = [k for k in required_cols if k not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {fp.name}")

    return df


def load_front_hdf5(
    c: Case,
    iso: float,
    *,
    dtype: str | None = "float32",
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    return load_front_csv(c, iso, dtype=dtype, required_cols=required_cols)

def load_fronts(
    c: Case,
    isolevels: list[float],
    **kwargs,
) -> dict[float, pd.DataFrame]:
    return {float(iso): load_front_csv(c, float(iso), **kwargs) for iso in isolevels}
