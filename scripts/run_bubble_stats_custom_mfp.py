#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lyman_alpha.bubble_stats_custom_mfp import histogram, sample_mfp_distances  # noqa: E402
from lyman_alpha.data import load_ionized_fraction, parse_redshift  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run custom MFP bubble-size statistics on one snapshot.")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--threshold", type=float, default=0.5, help="Ionized if fion >= threshold.")
    p.add_argument("--n-rays", type=int, default=5000)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--n-grid", type=int, default=200)
    p.add_argument("--box-size", type=float, default=200.0, help="Comoving Mpc/h")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fion = load_ionized_fraction(args.snapshot, n_grid=args.n_grid, memmap=False)
    mask = fion >= args.threshold
    if not np.any(mask):
        raise SystemExit("No ionized cells in this snapshot at chosen threshold.")

    distances = sample_mfp_distances(
        mask,
        n_rays=args.n_rays,
        box_size_mpc_h=args.box_size,
        seed=args.seed,
    )
    centers, pdf = histogram(distances, bins=args.bins)

    z = parse_redshift(args.snapshot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bubble_stats_z{z:.4f}_thr{args.threshold:.2f}_rays{args.n_rays}"
    np.savez(
        args.output_dir / f"{stem}.npz",
        redshift=z,
        threshold=args.threshold,
        n_rays=args.n_rays,
        distances_mpc_h=distances,
        hist_centers_mpc_h=centers,
        hist_pdf=pdf,
    )

    summary = args.output_dir / f"{stem}.txt"
    summary.write_text(
        "\n".join(
            [
                f"snapshot={args.snapshot}",
                f"redshift={z:.4f}",
                f"threshold={args.threshold}",
                f"n_rays={args.n_rays}",
                f"mean_distance_mpc_h={float(np.mean(distances)):.4f}",
                f"median_distance_mpc_h={float(np.median(distances)):.4f}",
                f"p90_distance_mpc_h={float(np.quantile(distances, 0.9)):.4f}",
            ]
        )
        + "\n"
    )

    print(f"Wrote: {args.output_dir / f'{stem}.npz'}")
    print(f"Wrote: {summary}")


if __name__ == "__main__":
    main()
