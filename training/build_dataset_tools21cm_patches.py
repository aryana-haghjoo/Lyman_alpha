#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lyman_alpha.data import list_snapshots, load_ionized_fraction, parse_redshift  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build large patch-level ML dataset with tools21cm MFP targets.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=Path("training/data/bubble_size_dataset_patches.csv"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--n-grid", type=int, default=200)
    p.add_argument("--box-size", type=float, default=200.0)
    p.add_argument("--hist-bins", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--patches-per-snapshot", type=int, default=16)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_features(fion_patch: np.ndarray, threshold: float, hist_bins: int) -> dict[str, float]:
    flat = fion_patch.ravel()
    hist, _ = np.histogram(flat, bins=hist_bins, range=(0.0, 1.0), density=True)
    out = {
        "fion_mean": float(np.mean(flat)),
        "fion_std": float(np.std(flat)),
        "fion_p10": float(np.quantile(flat, 0.10)),
        "fion_p25": float(np.quantile(flat, 0.25)),
        "fion_p50": float(np.quantile(flat, 0.50)),
        "fion_p75": float(np.quantile(flat, 0.75)),
        "fion_p90": float(np.quantile(flat, 0.90)),
        "ionized_volume_frac": float(np.mean(flat >= threshold)),
    }
    for i, v in enumerate(hist):
        out[f"fion_hist_{i:02d}"] = float(v)
    return out


def sample_patch(fion: np.ndarray, patch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int, int]]:
    n = fion.shape[0]
    if patch_size > n:
        raise ValueError(f"patch_size={patch_size} cannot exceed grid size {n}")
    i = int(rng.integers(0, n - patch_size + 1))
    j = int(rng.integers(0, n - patch_size + 1))
    k = int(rng.integers(0, n - patch_size + 1))
    return fion[i : i + patch_size, j : j + patch_size, k : k + patch_size], (i, j, k)


def main() -> None:
    args = parse_args()
    try:
        import tools21cm as t2c
    except Exception as exc:
        raise SystemExit("tools21cm is required. Install with: python -m pip install tools21cm") from exc

    rng = np.random.default_rng(args.seed)
    snapshots = list_snapshots(args.data_dir)
    if args.max_files:
        snapshots = snapshots[: args.max_files]
    if not snapshots:
        raise SystemExit(f"No snapshots found in: {args.data_dir}")

    rows: list[dict[str, float | int | str]] = []
    patch_box = args.box_size * (args.patch_size / args.n_grid)

    for i_snap, snap in enumerate(snapshots, start=1):
        print(f"[{i_snap}/{len(snapshots)}] {snap.name}")
        fion = load_ionized_fraction(snap, n_grid=args.n_grid, memmap=False).astype(np.float32)
        z = parse_redshift(snap)

        for pidx in range(args.patches_per_snapshot):
            patch, (x0, y0, z0) = sample_patch(fion, patch_size=args.patch_size, rng=rng)
            radii, pdf = t2c.bubble_stats.mfp(
                patch,
                xth=args.threshold,
                boxsize=patch_box,
                iterations=args.iterations,
                verbose=False,
            )
            radii = np.asarray(radii, dtype=np.float32)
            pdf = np.asarray(pdf, dtype=np.float32)
            peak_radius = float(radii[int(np.argmax(pdf))])

            row: dict[str, float | int | str] = {
                "snapshot": snap.name,
                "redshift": float(z),
                "patch_id": int(pidx),
                "patch_x0": int(x0),
                "patch_y0": int(y0),
                "patch_z0": int(z0),
                "patch_size": int(args.patch_size),
                "target_peak_radius_mpc_h": peak_radius,
            }
            row.update(make_features(patch, threshold=args.threshold, hist_bins=args.hist_bins))
            rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

