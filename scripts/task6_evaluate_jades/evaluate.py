#!/usr/bin/env /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3
"""
Task 6: Apply the trained Task 4 model to real JADES z=7-8 LAE spectra
and recover the bubble size distribution.

Pipeline:
  1. Load JADES z=7-8 spectra (NIRSpec prism, observed frame)
  2. Convert to rest frame, resample to simulation wavelength grid
  3. Normalise continuum → apply training normalisation stats
  4. Run through Task 4 CNN → predicted bubble size per galaxy
  5. Save predictions for notebook visualisation

Usage (from repo root):
    /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3 \\
        scripts/task6_evaluate_jades/evaluate.py
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
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reuse the model class from Task 4
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "task4_train_mock_lines"))
from train_model import LyaCNN  # noqa: E402

LYA_REST  = 1216.0   # Å
CONT_LO   = 1260.0   # Å  — continuum normalisation window (same as training)
CONT_HI   = 1310.0   # Å
UM_TO_AA  = 1e4


# ── JADES helpers ─────────────────────────────────────────────────────────

def build_jades_index(jades_dir: Path) -> dict[int, Path]:
    paths = glob.glob(str(jades_dir / "**" / "*prism*x1d*.fits"), recursive=True)
    out = {}
    for p in paths:
        try:
            obj_id = int(Path(p).stem.split("_")[4].split("-")[-1])
            out[obj_id] = Path(p)
        except (IndexError, ValueError):
            pass
    return out


def load_jades_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    try:
        with fits.open(path) as h:
            data = h["EXTRACT5PIX1D"].data
            wav  = np.array(data["WAVELENGTH"], dtype=float) * UM_TO_AA
            flux = np.array(data["FLUX"],       dtype=float)
            err  = np.array(data["FLUX_ERR"],   dtype=float)
        flux[(err <= 0) | ~np.isfinite(flux)] = np.nan
        return wav, flux
    except Exception:
        return None, None


def preprocess_spectrum(
    wav_obs: np.ndarray,
    flux: np.ndarray,
    z_obj: float,
    wav_grid: np.ndarray,
) -> np.ndarray | None:
    """
    Convert a JADES observed spectrum to the simulation rest-frame grid,
    continuum-normalise it (same way training templates were normalised),
    and return a float32 array of shape (N_wav,).
    Returns None if the spectrum cannot be reliably preprocessed.
    """
    wav_rest = wav_obs / (1.0 + z_obj)
    finite   = np.isfinite(flux)

    if finite.sum() < 50:
        return None
    if wav_rest[finite].min() > wav_grid.min() + 5:
        return None
    if wav_rest[finite].max() < wav_grid.max() - 5:
        return None

    interp_fn = interp1d(
        wav_rest[finite], flux[finite],
        kind="linear", bounds_error=False, fill_value=np.nan,
    )
    flux_grid = interp_fn(wav_grid).astype(np.float32)

    # Continuum normalisation
    cont_mask = (wav_grid >= CONT_LO) & (wav_grid <= CONT_HI)
    cont_vals = flux_grid[cont_mask]
    cont_vals = cont_vals[np.isfinite(cont_vals)]
    if len(cont_vals) < 5:
        return None
    cont_level = float(np.median(cont_vals))
    if cont_level <= 0:
        return None

    flux_norm = flux_grid / cont_level
    flux_norm = np.where(np.isfinite(flux_norm), flux_norm, 0.0)
    return flux_norm


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jades-dir",   type=Path,
                   default=Path("/home/aryana/Documents/GitHub/multimodal_superresolution/data/JADES/DR4"))
    p.add_argument("--model-dir",   type=Path,
                   default=PROJECT_ROOT / "results/task4_train_mock_lines")
    p.add_argument("--task1-dir",   type=Path,
                   default=PROJECT_ROOT / "results/task1_transmission_to_bubble_size")
    p.add_argument("--output-dir",  type=Path,
                   default=PROJECT_ROOT / "results/task6_evaluate_jades")
    p.add_argument("--z-min",       type=float, default=7.0)
    p.add_argument("--z-max",       type=float, default=8.0)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────
    norm  = np.load(args.model_dir / "normalisation_stats.npz")
    mu    = norm["spec_mu"].astype(np.float32)    # (1, N_wav)
    sigma = norm["spec_sigma"].astype(np.float32)
    z_mu  = float(norm["z_mu"])
    z_sig = float(norm["z_sigma"])

    n_wav = mu.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LyaCNN(n_wav=n_wav).to(device)
    model.load_state_dict(torch.load(
        args.model_dir / "best_model.pt", map_location=device
    ))
    model.eval()
    print(f"Model loaded ({n_wav} wav bins, device={device})")

    # ── load wavelength grid from training dataset ────────────────────────
    ds = np.load(sorted(args.task1_dir.glob("dataset_z*.npz"))[0])
    wav_grid = ds["wav"].astype(np.float64)   # (N_wav,) rest-frame Å
    sim_bubble_size = ds["bubble_size"]       # RT cube MFP for comparison
    print(f"Wavelength grid: {wav_grid.min():.1f}–{wav_grid.max():.1f} Å ({len(wav_grid)} bins)")

    # ── load JADES catalog and index ─────────────────────────────────────
    cat = Table.read(args.jades_dir / "Combined_DR4_external_v1.2.1.fits")
    z_spec = np.array(cat["z_Spec"],      dtype=float)
    flag   = np.array(cat["z_Spec_flag"], dtype=str)
    nid    = np.array(cat["NIRSpec_ID"],  dtype=int)

    eval_mask = (z_spec >= args.z_min) & (z_spec < args.z_max)
    eval_cat  = cat[eval_mask]
    print(f"JADES objects at z={args.z_min}–{args.z_max}: {eval_mask.sum()}")

    id_to_path = build_jades_index(args.jades_dir)

    # ── preprocess and predict ────────────────────────────────────────────
    pred_bubble = []
    true_z      = []
    obj_ids     = []
    skipped     = 0

    for row in eval_cat:
        obj_id = int(row["NIRSpec_ID"])
        z_obj  = float(row["z_Spec"])

        if obj_id not in id_to_path:
            skipped += 1
            continue

        wav_obs, flux = load_jades_spectrum(id_to_path[obj_id])
        if wav_obs is None:
            skipped += 1
            continue

        x = preprocess_spectrum(wav_obs, flux, z_obj, wav_grid)
        if x is None:
            skipped += 1
            continue

        # Apply training normalisation
        x_norm = (x - mu[0]) / sigma[0]  # (N_wav,)

        # Normalise redshift
        z_norm_val = np.float32((z_obj - z_mu) / z_sig)

        with torch.no_grad():
            spec_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)
            z_t    = torch.tensor([[z_norm_val]], device=device).squeeze(1)
            log_pred = model(spec_t, z_t).cpu().item()

        bubble_pred = float(np.expm1(log_pred))
        pred_bubble.append(bubble_pred)
        true_z.append(z_obj)
        obj_ids.append(obj_id)

    pred_bubble = np.array(pred_bubble, dtype=np.float32)
    true_z      = np.array(true_z,      dtype=np.float32)
    obj_ids     = np.array(obj_ids,     dtype=np.int32)

    print(f"\nProcessed: {len(pred_bubble)}  skipped: {skipped}")
    print(f"Predicted bubble size: min={pred_bubble.min():.2f}  "
          f"median={np.median(pred_bubble):.2f}  "
          f"max={pred_bubble.max():.2f}  Mpc/h")

    np.savez(
        args.output_dir / "jades_predictions.npz",
        pred_bubble_size = pred_bubble,
        z_spec           = true_z,
        obj_ids          = obj_ids,
        sim_bubble_size  = sim_bubble_size,
        wav_grid         = wav_grid,
    )
    print(f"Saved → {args.output_dir / 'jades_predictions.npz'}")


if __name__ == "__main__":
    main()
