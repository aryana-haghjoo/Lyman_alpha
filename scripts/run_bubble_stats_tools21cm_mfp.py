#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lyman_alpha.data import load_ionized_fraction, parse_redshift  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tools21cm MFP bubble-size statistics on one snapshot.")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--threshold", type=float, default=0.5, help="Ionization threshold used internally by tools21cm.")
    p.add_argument("--n-grid", type=int, default=200)
    p.add_argument("--box-size", type=float, default=200.0, help="Comoving Mpc/h")
    p.add_argument("--iterations", type=int, default=200000, help="Number of random rays/samples for MFP.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tools21cm as t2c
    except Exception as exc:
        raise SystemExit(
            "tools21cm is not installed in this environment. "
            "Activate ly_a and run: pip install tools21cm"
        ) from exc

    fion = load_ionized_fraction(args.snapshot, n_grid=args.n_grid, memmap=False).astype(np.float32)

    if args.verbose:
        print("Running tools21cm.bubble_stats.mfp ...")

    radii, pdf = t2c.bubble_stats.mfp(
        fion,
        xth=args.threshold,
        boxsize=args.box_size,
        iterations=args.iterations,
        verbose=args.verbose,
    )
    radii = np.asarray(radii, dtype=np.float32)
    pdf = np.asarray(pdf, dtype=np.float32)

    z = parse_redshift(args.snapshot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"bubble_stats_tools21cm_mfp_z{z:.4f}"
        f"_thr{args.threshold:.2f}"
        f"_iter{args.iterations}"
    )
    np.savez(
        args.output_dir / f"{stem}.npz",
        snapshot=str(args.snapshot),
        redshift=z,
        threshold=args.threshold,
        iterations=args.iterations,
        radii_mpc_h=radii,
        pdf=pdf,
    )

    summary = args.output_dir / f"{stem}.txt"
    peak_i = int(np.argmax(pdf)) if pdf.size else -1
    peak_radius = float(radii[peak_i]) if peak_i >= 0 else float("nan")
    summary.write_text(
        "\n".join(
            [
                f"snapshot={args.snapshot}",
                f"redshift={z:.4f}",
                f"threshold={args.threshold}",
                f"iterations={args.iterations}",
                f"n_bins={int(radii.size)}",
                f"peak_radius_mpc_h={peak_radius:.4f}",
            ]
        )
        + "\n"
    )

    print(f"Wrote: {args.output_dir / f'{stem}.npz'}")
    print(f"Wrote: {summary}")


if __name__ == "__main__":
    main()

