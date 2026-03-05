from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np


FIELD_INDEX = {
    "rho": 0,
    "temp": 1,
    "temp_a15": 2,
    "f_H1": 3,
    "f_He1": 4,
    "f_H2": 5,
    "f_He2": 6,
    "f_He3": 7,
    "gamma_H1_tot_sizeof": 8,
    "gamma_He1_tot_sizeof": 9,
    "mfp": 10,
    "mfp_912": 11,
    "heat_rate": 12,
    "cool_rate": 13,
    "avg_nh1": 14,
    "fion": 15,
    "tcross": 16,
    "zreion_out": 17,
    "fion_max": 18,
    "fion_HeIII": 19,
    "gamma_He2_tot_sizeof": 20,
}


def parse_redshift(path: Path) -> float:
    match = re.search(r"z=([0-9]+\.[0-9]+)", path.name)
    if not match:
        raise ValueError(f"Could not parse redshift from filename: {path}")
    return float(match.group(1))


def list_snapshots(data_dir: str | Path) -> list[Path]:
    base = Path(data_dir)
    snapshots = [p for p in base.iterdir() if p.is_file() and p.name.startswith("gas_z=")]
    return sorted(snapshots, key=parse_redshift)


def load_rt_cube(
    path: str | Path,
    n_grid: int = 200,
    n_fields: int = 21,
    dtype: np.dtype = np.float32,
    memmap: bool = False,
) -> np.ndarray:
    """
    Load one RT snapshot as shape (N, N, N, n_fields).

    File format: 1 header float followed by N^3*n_fields float32 values.
    """
    path = Path(path)
    expected_floats = 1 + n_grid * n_grid * n_grid * n_fields
    expected_bytes = expected_floats * np.dtype(dtype).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected file size for {path}. "
            f"Expected {expected_bytes} bytes, found {actual_bytes}."
        )

    if memmap:
        return np.memmap(
            path, dtype=dtype, mode="r", offset=np.dtype(dtype).itemsize, shape=(n_grid, n_grid, n_grid, n_fields)
        )

    raw = np.fromfile(path, dtype=dtype)
    return raw[1:].reshape((n_grid, n_grid, n_grid, n_fields))


def get_field(cube: np.ndarray, field: str) -> np.ndarray:
    if field not in FIELD_INDEX:
        raise KeyError(f"Unknown field '{field}'. Valid fields: {sorted(FIELD_INDEX.keys())}")
    return cube[..., FIELD_INDEX[field]]


def load_ionized_fraction(path: str | Path, n_grid: int = 200, memmap: bool = True) -> np.ndarray:
    cube = load_rt_cube(path, n_grid=n_grid, memmap=memmap)
    return get_field(cube, "fion")


def mean_ionized_fraction(paths: Iterable[Path], n_grid: int = 200) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for p in paths:
        z = parse_redshift(p)
        fion = load_ionized_fraction(p, n_grid=n_grid, memmap=True)
        output.append((z, float(fion.mean())))
    return output

