#!/usr/bin/env /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3
"""
Prepare per-sightline dataset for Task 1: transmission curve → bubble size.

Target (bubble_size_mpc_h): per-halo local MFP computed from the RT cube.
For each unique halo position we cast N_RAYS random rays through the fion
field at 1 Mpc/h steps (same methodology as tools21cm ray-casting) and
average the distances to the first neutral voxel (fion < ION_THRESHOLD).

The same MFP value is assigned to all sightlines from the same halo (bubble
size is a property of the halo's location, not the sightline direction).
Halos whose starting RT voxel is already neutral get MFP = 0.

Multi-redshift ready: for every redshift that has BOTH a Lya_transmission
file AND an RT cube, one dataset_z*.npz is saved under the output directory.
Adding new redshifts only requires placing new data files; re-run this script
and train_model.py will pick them all up automatically.

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

from lyman_alpha.data import load_rt_cube, get_field  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────
N_SKEWERS        = 10_000
N_WAV            = 2_000
CELL_SIZE_CMPC_H = 0.0244141   # Mpc/h — fine simulation grid cell
N_GRID           = 200
BOX_SIZE         = 200.0       # Mpc/h
RT_CELL          = BOX_SIZE / N_GRID  # 1.0 Mpc/h per RT voxel

N_RAYS           = 50          # random rays per halo for local MFP
MAX_STEPS        = 300         # max RT voxels to march (~300 Mpc/h)
ION_THRESHOLD    = 0.5         # fion < this → neutral

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


def find_redshift_pairs(
    lya_dir: Path, rt_dir: Path
) -> list[tuple[float, Path, Path]]:
    lya = {
        parse_redshift_from_filename(p.name): p
        for p in lya_dir.glob("Lya_transmission.z=*")
    }
    rt = {
        parse_redshift_from_filename(p.name): p
        for p in rt_dir.iterdir()
        if p.is_file() and p.name.startswith("gas_z=")
    }
    shared = sorted(set(lya) & set(rt))
    if not shared:
        raise RuntimeError("No redshifts with both transmission + RT cube.")
    print(f"Found {len(shared)} redshift(s): {shared}")
    return [(z, lya[z], rt[z]) for z in shared]


def load_params(lya_dir: Path, z: float) -> np.ndarray:
    path = lya_dir / f"sl_params_z={z:07.4f}"
    p = np.fromfile(path, dtype=PARAMS_DTYPE)
    if len(p) != N_SKEWERS:
        raise ValueError(f"Expected {N_SKEWERS} records, got {len(p)}")
    return p


def load_transmission(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != N_SKEWERS * N_WAV:
        raise ValueError(f"Unexpected transmission size {raw.size}")
    return raw.reshape(N_SKEWERS, N_WAV)


def load_wavelengths(lya_dir: Path) -> np.ndarray:
    wav = np.fromfile(lya_dir / "wav_out", dtype=np.float32)
    if wav.size != N_WAV:
        raise ValueError(f"Expected {N_WAV} wavelength bins")
    return wav


# ── multi-ray MFP from RT cube ─────────────────────────────────────────────

def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random unit vectors uniformly distributed on the sphere."""
    v = rng.standard_normal((n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def compute_multi_ray_mfp(
    fion: np.ndarray,
    params: np.ndarray,
    rng: np.random.Generator,
    n_rays: int = N_RAYS,
    n_steps: int = MAX_STEPS,
    threshold: float = ION_THRESHOLD,
) -> np.ndarray:
    """
    For each unique halo (grouped by start_inds), cast n_rays random rays
    through the RT cube at 1 Mpc/h steps and average the distances to the
    first neutral voxel (fion < threshold).

    Using random rays rather than the specific sightline direction gives a
    stable per-halo bubble size that is not polluted by angle noise — the
    same physical quantity the CNN should learn to predict.

    Returns float32 array of shape (N_SKEWERS,) in Mpc/h.
    Same value for all sightlines from the same halo.
    """
    # Group sightlines by unique halo position (same start_inds = same halo)
    unique_pos, inverse = np.unique(
        params["start_inds"], axis=0, return_inverse=True
    )
    n_halos = len(unique_pos)
    halo_mfp = np.zeros(n_halos, dtype=np.float32)

    start_pos_all = unique_pos.astype(np.float32) * CELL_SIZE_CMPC_H  # (n_halos, 3)
    steps = np.arange(1, n_steps + 1, dtype=np.float32)               # (n_steps,)

    for i in range(n_halos):
        pos0 = start_pos_all[i]

        # Cast n_rays random rays for every halo — no starting-voxel gate.
        # Halos in neutral regions naturally get small MFP (rays hit neutral
        # gas immediately); halos deep in bubbles get large MFP.
        dirs = random_unit_vectors(rng, n_rays)
        pos  = pos0[None, None, :] + steps[None, :, None] * dirs[:, None, :]
        vox  = (pos / RT_CELL).astype(np.int32) % N_GRID
        fion_ray = fion[vox[..., 0], vox[..., 1], vox[..., 2]]  # (n_rays, n_steps)

        neutral     = fion_ray < threshold
        has_neutral = neutral.any(axis=1)
        first_step  = np.argmax(neutral, axis=1) + 1
        first_step[~has_neutral] = n_steps

        halo_mfp[i] = float(first_step.mean()) * RT_CELL

        if (i + 1) % 100 == 0:
            print(f"  halo {i+1:4d}/{n_halos}  mfp={halo_mfp[i]:.1f} Mpc/h")

    print(f"\n  {n_halos} unique halos — no zeros forced")
    print(f"  MFP: min={halo_mfp.min():.1f}  median={np.median(halo_mfp):.1f}  "
          f"mean={halo_mfp.mean():.1f}  max={halo_mfp.max():.1f}  Mpc/h")

    return halo_mfp[inverse]  # broadcast per-halo value to all sightlines


# ── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lya-dir", type=Path,
                   default=PROJECT_ROOT / "data/for_aryana/late_end_early_start/Lya_transmission")
    p.add_argument("--rt-dir",  type=Path,
                   default=PROJECT_ROOT / "data/for_aryana/late_end_early_start/RTcubes")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results/task1_transmission_to_bubble_size")
    p.add_argument("--threshold", type=float, default=ION_THRESHOLD)
    p.add_argument("--n-rays",    type=int,   default=N_RAYS)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wav_raw   = load_wavelengths(args.lya_dir)
    wav_order = np.argsort(wav_raw)
    wav       = wav_raw[wav_order]

    rng   = np.random.default_rng(args.seed)
    pairs = find_redshift_pairs(args.lya_dir, args.rt_dir)

    for z, lya_path, rt_path in pairs:
        print(f"\n── z={z:.4f} ──────────────────────────────────────────")

        out_path = args.output_dir / f"dataset_z{z:.4f}.npz"
        if out_path.exists():
            print(f"  already exists, skipping.")
            continue

        params       = load_params(args.lya_dir, z)
        transmission = load_transmission(lya_path)[:, wav_order]

        print(f"  loading RT cube fion …")
        fion = get_field(load_rt_cube(rt_path, memmap=True), "fion")
        print(f"  fion global mean = {fion.mean():.3f}")

        print(f"  computing multi-ray MFP "
              f"({args.n_rays} rays/halo, max={MAX_STEPS} Mpc/h) …")
        bubble_r = compute_multi_ray_mfp(
            fion, params, rng,
            n_rays=args.n_rays,
            threshold=args.threshold,
        )

        np.savez(
            out_path,
            transmission=transmission,
            wav=wav,
            bubble_size=bubble_r,
            halo_mass=params["halo_mass"],
            redshift=np.float32(z),
            threshold=np.float32(args.threshold),
            n_rays=np.int32(args.n_rays),
        )
        print(f"  saved → {out_path}")
        print(f"  bubble_size: zeros={(bubble_r==0).mean()*100:.1f}%  "
              f"median(>0)={np.median(bubble_r[bubble_r>0]):.1f}  "
              f"max={bubble_r.max():.1f}  Mpc/h")

    print("\nDone.")


if __name__ == "__main__":
    main()
