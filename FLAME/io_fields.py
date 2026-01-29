from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pandas as pd
import numpy as np

from .io_fronts import Case, folder

FIELD_HDF_KEY = "field"


def field_filename(c: Case) -> str:
    tag = "post_" if c.post else ""
    return f"extracted_field_{tag}{c.time_step}.hdf5"


def field_path(c: Case) -> Path:
    return folder(c) / field_filename(c)


def load_field_csv(
    c: Case,
    *,
    dtype: str | None = "float32",
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    fp = field_path(c)
    if fp.exists():
        df = pd.read_hdf(fp, key=FIELD_HDF_KEY)
    else:
        csv_fp = fp.with_suffix(".csv")
        if csv_fp.exists():
            df = pd.read_csv(csv_fp)
        else:
            raise FileNotFoundError(f"Missing field file: {fp}")

    if dtype is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].astype(dtype)

    if required_cols:
        missing = [k for k in required_cols if k not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {fp.name}")

    return df


def load_field_hdf5(
    c: Case,
    *,
    dtype: str | None = "float32",
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    return load_field_csv(c, dtype=dtype, required_cols=required_cols)


def make_case_with_base_dir(c: Case, base_dir: Path) -> Case:
    return replace(c, base_dir=base_dir)
