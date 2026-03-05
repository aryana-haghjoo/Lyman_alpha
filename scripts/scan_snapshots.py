#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lyman_alpha.data import list_snapshots, mean_ionized_fraction  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute mean ionized fraction vs redshift.")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing gas_z=* snapshot files.")
    p.add_argument("--output", type=Path, default=Path("results/ionization_history.csv"))
    p.add_argument("--max-files", type=int, default=None, help="Process only first N snapshots (for quick tests).")
    p.add_argument("--n-grid", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = list_snapshots(args.data_dir)
    if args.max_files:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"No snapshot files found in {args.data_dir}")

    stats = mean_ionized_fraction(paths, n_grid=args.n_grid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["redshift", "mean_fion"])
        for z, m in stats:
            writer.writerow([f"{z:.4f}", f"{m:.8f}"])

    z_arr = [z for z, _ in stats]
    m_arr = [m for _, m in stats]
    best_i = min(range(len(m_arr)), key=lambda i: abs(m_arr[i] - 0.5))
    print(f"Wrote {len(stats)} rows to {args.output}")
    print(f"Closest to x_HII=0.5: z={z_arr[best_i]:.4f}, mean_fion={m_arr[best_i]:.4f}")


if __name__ == "__main__":
    main()

