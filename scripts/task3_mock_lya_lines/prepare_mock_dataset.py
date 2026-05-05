#!/usr/bin/env /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3
"""
Task 3: Generate mock z~7.5 Lya lines.

    mock_obs(λ) = F_intrinsic(λ) × T_IGM(λ)

For each simulation sightline we randomly draw one DEIMOS intrinsic Lya
template, resample it onto the simulation wavelength grid (1166–1350 Å rest
frame), normalise the continuum to 1, then multiply by the simulation
transmission curve T(λ).

The result is a realistic mock "observed" Lya line at z~7.5 — what an
observer would actually see through a spectrograph, as opposed to the raw
transmission curve.  The bubble_size labels and wavelength grid are
inherited unchanged from the Task 1 dataset.

Multi-redshift ready: processes every dataset_z*.npz it finds.

Usage (screen session, from repo root):
    /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3 \\
        scripts/task3_mock_lya_lines/prepare_mock_dataset.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import numpy as np
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LYA_REST       = 1216.0   # Å
CONT_LO        = 1260.0   # Å  — continuum normalisation window
CONT_HI        = 1310.0   # Å
MIN_CONT_BINS  = 10       # require at least this many valid bins in cont window


# ── DEIMOS template loading ────────────────────────────────────────────────

def build_id_to_path(spec_dir: Path) -> dict[int, Path]:
    """Map object ID → FITS path from spec1d.MASK.SLITNO.ID.fits naming."""
    paths = glob.glob(str(spec_dir / "**" / "*.fits"), recursive=True)
    out = {}
    for p in paths:
        try:
            obj_id = int(Path(p).stem.split(".")[-1])
            out[obj_id] = Path(p)
        except ValueError:
            pass
    return out


def load_1d_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Load (wavelength_Å, flux) from a DEIMOS spec1d FITS file.
    Tries Horne-B (blue arm) first, then Horne-R (red arm).
    Returns (None, None) on failure.
    """
    with fits.open(path) as hdul:
        for ext_name in ("Horne-B", "Horne-R", "Bxspf-B", "Bxspf-R"):
            if ext_name not in [h.name for h in hdul]:
                continue
            data = hdul[ext_name].data
            wav  = data["LAMBDA"][0].astype(np.float64)
            flux = data["SPEC"][0].astype(np.float64)
            ivar = data["IVAR"][0].astype(np.float64)
            flux[ivar == 0] = np.nan
            if np.isfinite(flux).sum() > 50:
                return wav, flux
    return None, None


def make_templates(
    cat: Table,
    id_to_path: dict[int, Path],
    wav_grid: np.ndarray,
    z_max_template: float = 5.0,
    q_min: int = 3,
) -> np.ndarray:
    """
    Build normalised intrinsic templates on wav_grid for all secure LAEs.

    Normalisation: divide by median flux in the continuum window
    (CONT_LO – CONT_HI Å rest frame) so the continuum = 1.

    Returns float32 array of shape (N_templates, N_wav).
    Bad / low-S/N spectra are skipped.
    """
    good = cat[(cat["Q_flag"] >= q_min)
               & (cat["z_helio"] > 2.0)
               & (cat["z_helio"] <= z_max_template)]

    templates = []
    skipped   = 0

    for row in good:
        obj_id = int(row["ID"])
        if obj_id not in id_to_path:
            skipped += 1
            continue

        wav_obs, flux = load_1d_spectrum(id_to_path[obj_id])
        if wav_obs is None:
            skipped += 1
            continue

        z = float(row["z_helio"])
        wav_rest = wav_obs / (1.0 + z)

        # Need coverage over our simulation grid
        if wav_rest[np.isfinite(flux)].min() > wav_grid.min() + 5:
            skipped += 1
            continue
        if wav_rest[np.isfinite(flux)].max() < wav_grid.max() - 5:
            skipped += 1
            continue

        # Interpolate onto simulation wavelength grid
        valid = np.isfinite(flux)
        if valid.sum() < 100:
            skipped += 1
            continue

        interp = interp1d(
            wav_rest[valid], flux[valid],
            kind="linear", bounds_error=False, fill_value=np.nan,
        )
        flux_grid = interp(wav_grid).astype(np.float32)

        # Normalise by continuum (red side, IGM transparent at these z)
        cont_mask = (wav_grid >= CONT_LO) & (wav_grid <= CONT_HI)
        cont_vals = flux_grid[cont_mask]
        cont_vals = cont_vals[np.isfinite(cont_vals)]
        if len(cont_vals) < MIN_CONT_BINS:
            skipped += 1
            continue
        cont_level = float(np.median(cont_vals))
        if cont_level <= 0:
            skipped += 1
            continue

        flux_norm = flux_grid / cont_level

        # Replace remaining NaNs with 0 (outside coverage → no flux)
        flux_norm = np.where(np.isfinite(flux_norm), flux_norm, 0.0)

        # Sanity: at least some positive flux near Lya
        lya_mask = (wav_grid > 1210) & (wav_grid < 1240)
        lya_peak = flux_norm[lya_mask].max()
        if lya_peak < 0.1:
            skipped += 1
            continue

        # Reject extreme emitters (Lya peak > 50x continuum — unusable shape)
        if lya_peak > 50:
            skipped += 1
            continue

        templates.append(flux_norm)

    print(f"  Templates built: {len(templates)}  skipped: {skipped}")
    return np.array(templates, dtype=np.float32)


# ── mock spectrum generation ───────────────────────────────────────────────

def make_mock_spectra(
    transmission: np.ndarray,   # (N_sightlines, N_wav)
    templates: np.ndarray,      # (N_templates, N_wav)
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For each sightline, draw one random template and multiply by transmission.

        mock_obs(λ) = F_intrinsic(λ) × T_IGM(λ)

    Returns float32 array of shape (N_sightlines, N_wav).
    """
    N = len(transmission)
    idx = rng.integers(0, len(templates), size=N)
    mock = templates[idx] * transmission        # broadcast: (N, N_wav)
    return mock.astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task1-dir", type=Path,
                   default=PROJECT_ROOT / "results/task1_transmission_to_bubble_size")
    p.add_argument("--deimos-dir", type=Path,
                   default=PROJECT_ROOT / "data/deimos_deeper_than_deep")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results/task3_mock_lya_lines")
    p.add_argument("--z-max-template", type=float, default=5.0,
                   help="Max redshift for DEIMOS templates (IGM transparent below ~5)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── build DEIMOS templates ─────────────────────────────────────────────
    print("Loading DEIMOS catalog …")
    cat = Table.read(args.deimos_dir / "DEIMOS_EGS_UrbanoStawinski.fits")
    id_to_path = build_id_to_path(args.deimos_dir / "1dspec")
    print(f"  Catalog: {len(cat)} objects  |  FITS indexed: {len(id_to_path)}")

    # ── process each redshift dataset ─────────────────────────────────────
    datasets = sorted(args.task1_dir.glob("dataset_z*.npz"))
    if not datasets:
        raise FileNotFoundError(f"No dataset_z*.npz in {args.task1_dir}")
    print(f"\nFound {len(datasets)} Task 1 dataset(s).")

    for ds_path in datasets:
        tag = ds_path.stem.replace("dataset_", "")   # e.g. z7.4985
        out_path = args.output_dir / f"mock_{tag}.npz"

        if out_path.exists():
            print(f"\n{tag}: already exists, skipping.")
            continue

        print(f"\n── {tag} ──────────────────────────────────────────")
        d = np.load(ds_path)
        wav          = d["wav"]            # (N_wav,) ascending Å
        transmission = d["transmission"]   # (N, N_wav)
        bubble_size  = d["bubble_size"]    # (N,)
        halo_mass    = d["halo_mass"]      # (N,)
        redshift     = d["redshift"]       # scalar

        print(f"  Sightlines: {len(transmission)}  wav: {wav.min():.1f}–{wav.max():.1f} Å")

        print("  Building intrinsic templates …")
        templates = make_templates(cat, id_to_path, wav,
                                   z_max_template=args.z_max_template)
        if len(templates) == 0:
            print("  ERROR: no valid templates — skipping this redshift.")
            continue
        print(f"  Template shape: {templates.shape}  "
              f"continuum-normalised, on simulation grid")

        print("  Generating mock spectra (intrinsic × transmission) …")
        mock = make_mock_spectra(transmission, templates, rng)
        print(f"  Mock spectra shape: {mock.shape}")

        np.savez(
            out_path,
            mock_spectra=mock,
            transmission=transmission,
            wav=wav,
            bubble_size=bubble_size,
            halo_mass=halo_mass,
            redshift=redshift,
            n_templates=np.int32(len(templates)),
        )
        print(f"  Saved → {out_path}")
        print(f"  Mock flux: median={np.median(mock):.4f}  "
              f"max={mock.max():.4f}  "
              f"zeros={(mock == 0).mean()*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
