#!/usr/bin/env /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3
"""
Task 3: Generate mock z~7.5 Lya lines.

    mock_obs(λ) = F_intrinsic(λ) × T_IGM(λ)

Intrinsic templates come from JADES DR4 NIRSpec/prism LAEs at z=4–5.5
(IGM transparent at those redshifts, same instrument as the z~7.5
evaluation data → no resolution mismatch).

For each simulation sightline we randomly draw one template, resample it
onto the simulation wavelength grid (1166–1350 Å rest frame), normalise
the continuum to 1, then multiply by the simulation transmission curve.

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

LYA_REST      = 1216.0   # Å
CONT_LO       = 1260.0   # Å — continuum normalisation window
CONT_HI       = 1310.0   # Å
MIN_CONT_BINS = 10
UM_TO_AA      = 1e4      # 1 μm = 10 000 Å


# ── JADES template loading ─────────────────────────────────────────────────

def build_jades_index(jades_dir: Path) -> dict[int, Path]:
    """Map NIRSpec_ID → prism x1d FITS path."""
    paths = glob.glob(
        str(jades_dir / "**" / "*prism*x1d*.fits"), recursive=True
    )
    out = {}
    for p in paths:
        stem  = Path(p).stem   # e.g. hlsp_jades_jwst_nirspec_goods-s-deepjwst-00002787_clear-prism_v1.0_x1d
        parts = stem.split("_")
        try:
            obj_id = int(parts[4].split("-")[-1])
            out[obj_id] = Path(p)
        except (IndexError, ValueError):
            pass
    return out


def load_jades_spectrum(
    path: Path,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Load (wavelength_Å_obs, flux) from a JADES prism x1d FITS file.
    Uses the 5-pixel extraction (best S/N for point sources).
    Wavelengths in the file are in microns → converted to Å.
    """
    try:
        with fits.open(path) as hdul:
            data = hdul["EXTRACT5PIX1D"].data
            wav  = np.array(data["WAVELENGTH"], dtype=np.float64) * UM_TO_AA
            flux = np.array(data["FLUX"],       dtype=np.float64)
            err  = np.array(data["FLUX_ERR"],   dtype=np.float64)
        # mask bad pixels (zero or negative error)
        bad = (err <= 0) | ~np.isfinite(flux) | ~np.isfinite(err)
        flux[bad] = np.nan
        return wav, flux
    except Exception:
        return None, None


def make_jades_templates(
    cat: Table,
    id_to_path: dict[int, Path],
    wav_grid: np.ndarray,
    z_min: float = 4.0,
    z_max: float = 5.5,
    good_flags: tuple[str, ...] = ("A", "B"),
    max_lya_peak: float = 50.0,
) -> np.ndarray:
    """
    Build continuum-normalised Lya templates on wav_grid from JADES
    LAEs at z_min ≤ z ≤ z_max with secure redshifts.

    z=4–5.5 ensures Lya (1216 Å) falls within the NIRSpec prism range
    (≥0.6 μm) while the IGM is still largely transparent.

    Returns float32 array of shape (N_templates, N_wav).
    """
    z_spec = np.array(cat["z_Spec"],      dtype=float)
    flag   = np.array(cat["z_Spec_flag"], dtype=str)
    nid    = np.array(cat["NIRSpec_ID"],  dtype=int)

    sel = (
        (z_spec >= z_min) & (z_spec <= z_max)
        & np.isin(flag, list(good_flags))
    )
    selected = cat[sel]
    print(f"  Candidate templates (z={z_min}–{z_max}, flag {good_flags}): {sel.sum()}")

    templates = []
    skipped   = 0

    for row in selected:
        obj_id = int(row["NIRSpec_ID"])
        z      = float(row["z_Spec"])

        if obj_id not in id_to_path:
            skipped += 1
            continue

        wav_obs, flux = load_jades_spectrum(id_to_path[obj_id])
        if wav_obs is None:
            skipped += 1
            continue

        # Convert to rest frame
        wav_rest = wav_obs / (1.0 + z)

        # Need coverage over simulation grid
        finite = np.isfinite(flux)
        if finite.sum() < 50:
            skipped += 1
            continue
        if wav_rest[finite].min() > wav_grid.min() + 5:
            skipped += 1
            continue
        if wav_rest[finite].max() < wav_grid.max() - 5:
            skipped += 1
            continue

        # Interpolate onto simulation grid
        interp_fn  = interp1d(
            wav_rest[finite], flux[finite],
            kind="linear", bounds_error=False, fill_value=np.nan,
        )
        flux_grid = interp_fn(wav_grid).astype(np.float32)

        # Normalise by red-side continuum
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
        flux_norm = np.where(np.isfinite(flux_norm), flux_norm, 0.0)

        # Sanity: positive Lya flux
        lya_mask = (wav_grid > 1210) & (wav_grid < 1240)
        lya_peak = float(flux_norm[lya_mask].max())
        if lya_peak < 0.1:
            skipped += 1
            continue

        # Reject extreme outliers
        if lya_peak > max_lya_peak:
            skipped += 1
            continue

        templates.append(flux_norm)

    print(f"  Templates built: {len(templates)}  skipped: {skipped}")
    return np.array(templates, dtype=np.float32)


# ── mock spectrum generation ───────────────────────────────────────────────

def make_mock_spectra(
    transmission: np.ndarray,
    templates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For each sightline draw one random JADES template and multiply by T(λ).
        mock_obs(λ) = F_intrinsic(λ) × T_IGM(λ)
    Returns float32 (N_sightlines, N_wav).
    """
    idx  = rng.integers(0, len(templates), size=len(transmission))
    mock = templates[idx] * transmission
    return mock.astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task1-dir", type=Path,
                   default=PROJECT_ROOT / "results/task1_transmission_to_bubble_size")
    p.add_argument("--jades-dir", type=Path,
                   default=Path("/home/aryana/Documents/GitHub/multimodal_superresolution/data/JADES/DR4"))
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results/task3_mock_lya_lines")
    p.add_argument("--z-min-template", type=float, default=4.0)
    p.add_argument("--z-max-template", type=float, default=5.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── build JADES template index ────────────────────────────────────────
    print("Indexing JADES prism spectra …")
    id_to_path = build_jades_index(args.jades_dir)
    print(f"  Indexed {len(id_to_path)} prism x1d files")

    print("Loading JADES catalog …")
    cat = Table.read(args.jades_dir / "Combined_DR4_external_v1.2.1.fits")
    print(f"  {len(cat)} objects in catalog")

    # ── process each simulation redshift ──────────────────────────────────
    datasets = sorted(args.task1_dir.glob("dataset_z*.npz"))
    if not datasets:
        raise FileNotFoundError(f"No dataset_z*.npz in {args.task1_dir}")
    print(f"\nFound {len(datasets)} Task 1 dataset(s).")

    for ds_path in datasets:
        tag      = ds_path.stem.replace("dataset_", "")
        out_path = args.output_dir / f"mock_{tag}.npz"

        if out_path.exists():
            print(f"\n{tag}: already exists, skipping.")
            continue

        print(f"\n── {tag} ──────────────────────────────────────────")
        d = np.load(ds_path)
        wav          = d["wav"]
        transmission = d["transmission"]
        bubble_size  = d["bubble_size"]
        halo_mass    = d["halo_mass"]
        redshift     = d["redshift"]
        print(f"  Sightlines: {len(transmission)}  wav: {wav.min():.1f}–{wav.max():.1f} Å")

        print(f"  Building JADES templates (z={args.z_min_template}–{args.z_max_template}) …")
        templates = make_jades_templates(
            cat, id_to_path, wav,
            z_min=args.z_min_template,
            z_max=args.z_max_template,
        )
        if len(templates) == 0:
            print("  ERROR: no valid templates — skipping.")
            continue

        print(f"  Generating mock spectra …")
        mock = make_mock_spectra(transmission, templates, rng)

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
              f"max={mock.max():.4f}  zeros={(mock==0).mean()*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
