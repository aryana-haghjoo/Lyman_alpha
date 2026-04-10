#!/usr/bin/env /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3
"""
Prepare per-sightline dataset for Task 1: transmission curve → bubble size.

Target (bubble_size_mpc_h): derived directly from each transmission curve.
The Lya transmission rises from ~0 (blue of line centre) to ~1 (red wing)
at a wavelength that encodes the ionized bubble size via the Hubble flow:

    R_bubble [Mpc/h] ≈ (λ_edge - 1216 Å) * c / (1216 Å * H(z) / h)

where λ_edge is where T first exceeds T_THRESHOLD scanning red-ward from 1216 Å.
Sightlines with no detectable edge (galaxy fully outside any ionized region)
get R = 0.

This target is resolution-independent and directly observable, unlike
RT-cube-derived targets which are limited to the 1 Mpc/h voxel size.

For every redshift that has BOTH a Lya_transmission file AND an RT cube we
save one dataset_z*.npz under results/task1_transmission_to_bubble_size/.

Usage (screen session, from repo root):
    /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3 \\
        scripts/task1_transmission_to_bubble_size/prepare_dataset.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── constants ──────────────────────────────────────────────────────────────
N_SKEWERS = 10_000
N_WAV     = 2_000
LYA_REST  = 1216.0   # Å

# Bubble size conversion: R [Mpc/h] = delta_lambda [Å] * C_CONV
# delta_lambda = lambda_edge - LYA_REST
# R = delta_lambda * c / (LYA_REST * H(z)/h)
# At z~7.5: H(z)/h ≈ 950/0.7 ≈ 1357 km/s/(Mpc/h), c=3e5 km/s
# C_CONV = 3e5 / (1216 * 1357) ≈ 0.1815 (Mpc/h)/Å
C_CONV = 3e5 / (LYA_REST * 1357.0)   # (Mpc/h) per Å of edge shift

T_THRESHOLD = 0.5    # transmission level that marks the bubble edge

PARAMS_DTYPE = np.dtype([
    ("start_inds", "<i4", (3,)),
    ("angles",     "<f4", (2,)),
    ("halo_mass",  "<f4"),
])


# ── I/O helpers ───────────────────────────────────────────────────────────

def parse_redshift_from_filename(name: str) -> float:
    m = re.search(r"z=0?(\d+\.\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse redshift from: {name}")
    return float(m.group(1))


def find_lya_redshifts(lya_dir: Path) -> list[tuple[float, Path]]:
    files = {
        parse_redshift_from_filename(p.name): p
        for p in lya_dir.glob("Lya_transmission.z=*")
    }
    if not files:
        raise RuntimeError(f"No Lya_transmission files found in {lya_dir}")
    result = sorted(files.items())
    print(f"Found {len(result)} transmission redshift(s): {[z for z,_ in result]}")
    return result


def load_params(lya_dir: Path, z: float) -> np.ndarray:
    tag  = f"{z:07.4f}"
    path = lya_dir / f"sl_params_z={tag}"
    params = np.fromfile(path, dtype=PARAMS_DTYPE)
    if params.shape[0] != N_SKEWERS:
        raise ValueError(f"Expected {N_SKEWERS} records, got {params.shape[0]}")
    return params


def load_transmission(lya_path: Path) -> np.ndarray:
    raw = np.fromfile(lya_path, dtype=np.float32)
    if raw.size != N_SKEWERS * N_WAV:
        raise ValueError(f"Unexpected size {raw.size}")
    return raw.reshape(N_SKEWERS, N_WAV)


def load_wavelengths(lya_dir: Path) -> np.ndarray:
    wav = np.fromfile(lya_dir / "wav_out", dtype=np.float32)
    if wav.size != N_WAV:
        raise ValueError(f"Expected {N_WAV} bins, got {wav.size}")
    return wav


# ── bubble size from transmission edge ────────────────────────────────────

def compute_bubble_sizes(
    transmission: np.ndarray,
    wav: np.ndarray,
    threshold: float = T_THRESHOLD,
) -> np.ndarray:
    """
    For each sightline, find the first wavelength RED-WARD of Lya (>= LYA_REST)
    where T >= threshold.  Convert the offset Δλ = λ_edge − LYA_REST to a
    comoving bubble radius in Mpc/h via:

        R [Mpc/h] = Δλ [Å] × C_CONV

    Sightlines where T < threshold everywhere redward of Lya get R = 0
    (galaxy outside any detectable ionized region).
    """
    # wav is in descending order (1350→1166 Å); sort ascending before scanning
    order     = np.argsort(wav)               # ascending index map
    wav_asc   = wav[order]                    # (N_WAV,) ascending
    T_asc     = transmission[:, order]        # (N_SKEWERS, N_WAV) ascending

    # only scan wavelengths >= LYA_REST (scanning blue→red from line centre)
    red_mask  = wav_asc >= LYA_REST           # (N_WAV,) bool
    wav_red   = wav_asc[red_mask]             # (N_red,)  ascending from ~1216 Å
    T_red     = T_asc[:, red_mask]            # (N_SKEWERS, N_red)

    # first bin where T >= threshold scanning outward from 1216 Å
    above     = T_red >= threshold            # (N_SKEWERS, N_red)
    has_edge  = above.any(axis=1)             # (N_SKEWERS,)
    first_idx = np.argmax(above, axis=1)      # index of first True; 0 if none

    lam_edge  = np.where(has_edge, wav_red[first_idx], LYA_REST)
    delta_lam = np.maximum(lam_edge - LYA_REST, 0.0)
    bubble_r  = (delta_lam * C_CONV).astype(np.float32)
    return bubble_r


# ── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lya-dir", type=Path,
                   default=PROJECT_ROOT / "data/for_aryana/late_end_early_start/Lya_transmission")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results/task1_transmission_to_bubble_size")
    p.add_argument("--threshold", type=float, default=T_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wav_raw   = load_wavelengths(args.lya_dir)
    wav_order = np.argsort(wav_raw)          # sort ascending (1166→1350 Å)
    wav       = wav_raw[wav_order]
    pairs     = find_lya_redshifts(args.lya_dir)

    for z, lya_path in pairs:
        print(f"\n── z={z:.4f} ──────────────────────────────────────────")

        tag      = f"z{z:.4f}"
        out_path = args.output_dir / f"dataset_{tag}.npz"
        if out_path.exists():
            print(f"  already exists, skipping: {out_path}")
            continue

        params            = load_params(args.lya_dir, z)
        transmission_raw  = load_transmission(lya_path)
        transmission      = transmission_raw[:, wav_order]   # reorder columns ascending

        print(f"  computing bubble sizes from transmission edge …")
        bubble_r = compute_bubble_sizes(transmission, wav, threshold=args.threshold)

        np.savez(
            out_path,
            transmission=transmission,
            wav=wav,
            bubble_size=bubble_r,
            halo_mass=params["halo_mass"],
            redshift=np.float32(z),
            edge_threshold=np.float32(args.threshold),
        )
        print(f"  saved → {out_path}")
        print(f"  bubble size: zeros={( bubble_r==0).mean()*100:.1f}%  "
              f"median={np.median(bubble_r):.3f}  max={bubble_r.max():.3f}  Mpc/h")

    print("\nDone.")


if __name__ == "__main__":
    main()
