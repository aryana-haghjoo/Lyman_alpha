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
    p = argparse.ArgumentParser(description="Build ML dataset for bubble-size prediction using tools21cm targets.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=Path("training/data/bubble_size_dataset.csv"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--iterations", type=int, default=200000)
    p.add_argument("--n-grid", type=int, default=200)
    p.add_argument("--box-size", type=float, default=200.0)
    p.add_argument("--hist-bins", type=int, default=32)
    p.add_argument("--max-files", type=int, default=None)
    return p.parse_args()


def make_features(fion: np.ndarray, threshold: float, hist_bins: int) -> dict[str, float]:
    flat = fion.ravel()
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


def main() -> None:
    args = parse_args()
    try:
        import tools21cm as t2c
    except Exception as exc:
        raise SystemExit("tools21cm is required. Install with: python -m pip install tools21cm") from exc

    snapshots = list_snapshots(args.data_dir)
    if args.max_files:
        snapshots = snapshots[: args.max_files]
    if not snapshots:
        raise SystemExit(f"No snapshots found in: {args.data_dir}")

    rows: list[dict[str, float | str]] = []

    for i, snap in enumerate(snapshots, start=1):
        print(f"[{i}/{len(snapshots)}] {snap.name}")
        fion = load_ionized_fraction(snap, n_grid=args.n_grid, memmap=False).astype(np.float32)
        z = parse_redshift(snap)

        radii, pdf = t2c.bubble_stats.mfp(
            fion,
            xth=args.threshold,
            boxsize=args.box_size,
            iterations=args.iterations,
            verbose=False,
        )
        radii = np.asarray(radii, dtype=np.float32)
        pdf = np.asarray(pdf, dtype=np.float32)
        peak_i = int(np.argmax(pdf))
        peak_radius = float(radii[peak_i])

        row: dict[str, float | str] = {
            "snapshot": snap.name,
            "redshift": float(z),
            "target_peak_radius_mpc_h": peak_radius,
        }
        row.update(make_features(fion, threshold=args.threshold, hist_bins=args.hist_bins))
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

