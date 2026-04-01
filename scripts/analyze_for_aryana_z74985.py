from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_SKEWERS = 10_000
N_CELLS = 4_000
N_WAV = 2_000
CELL_SIZE_CMPC_H = 0.0244141


PARAMS_DTYPE = np.dtype(
    [
        ("start_inds", "<i4", (3,)),
        ("angles", "<f4", (2,)),
        ("halo_mass", "<f4"),
    ]
)


def load_params(path: Path) -> np.ndarray:
    params = np.fromfile(path, dtype=PARAMS_DTYPE)
    if params.shape[0] != N_SKEWERS:
        raise ValueError(f"Expected {N_SKEWERS} skewer records, found {params.shape[0]}")
    return params


def load_sightlines(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=np.float32)
    expected = N_SKEWERS * N_CELLS * 2
    if raw.size != expected:
        raise ValueError(f"Expected {expected} float32 values, found {raw.size}")
    sightlines = raw.reshape(N_SKEWERS, N_CELLS, 2)
    return sightlines[..., 0], sightlines[..., 1]


def load_transmission(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    expected = N_SKEWERS * N_WAV
    if raw.size != expected:
        raise ValueError(f"Expected {expected} float32 values, found {raw.size}")
    return raw.reshape(N_SKEWERS, N_WAV)


def load_wavelengths(path: Path) -> np.ndarray:
    wav = np.fromfile(path, dtype=np.float32)
    if wav.size != N_WAV:
        raise ValueError(f"Expected {N_WAV} wavelength bins, found {wav.size}")
    return wav


def make_angle_plot(params: np.ndarray, output_path: Path) -> dict[str, float]:
    theta = params["angles"][:, 0]
    phi = params["angles"][:, 1]
    cos_theta = np.cos(theta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(cos_theta, bins=30, density=True, color="#4C78A8", edgecolor="white")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel(r"$\cos(\theta)$")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title(r"Distribution of $\cos(\theta)$")

    axes[1].hist(phi, bins=30, density=True, color="#F58518", edgecolor="white")
    axes[1].axhline(1.0 / (2.0 * np.pi), color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"$\phi$ [rad]")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title(r"Distribution of $\phi$")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "cos_theta_mean": float(cos_theta.mean()),
        "cos_theta_std": float(cos_theta.std()),
        "phi_mean": float(phi.mean()),
        "phi_std": float(phi.std()),
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "phi_min": float(phi.min()),
        "phi_max": float(phi.max()),
    }


def make_profile_plot(
    x: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample_idx = rng.choice(values.shape[0], size=20, replace=False)
    mean_profile = values.mean(axis=0)
    p16 = np.percentile(values, 16, axis=0)
    p84 = np.percentile(values, 84, axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for idx in sample_idx:
        ax.plot(x, values[idx], color="#A0CBE8", alpha=0.35, linewidth=0.7)
    ax.plot(x, mean_profile, color="#1F77B4", linewidth=2.2, label="Mean")
    ax.fill_between(x, p16, p84, color="#1F77B4", alpha=0.18, label="16th-84th percentile")
    ax.set_xlabel(r"Distance from halo boundary [cMpc/$h$]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_start": float(mean_profile[0]),
        "mean_mid": float(mean_profile[len(mean_profile) // 2]),
        "mean_end": float(mean_profile[-1]),
        "global_min": float(values.min()),
        "global_max": float(values.max()),
    }


def make_transmission_plot(
    wav: np.ndarray,
    transmission: np.ndarray,
    output_path: Path,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample_idx = rng.choice(transmission.shape[0], size=20, replace=False)
    mean_spec = transmission.mean(axis=0)
    p16 = np.percentile(transmission, 16, axis=0)
    p84 = np.percentile(transmission, 84, axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for idx in sample_idx:
        ax.plot(wav, transmission[idx], color="#FFBE7D", alpha=0.35, linewidth=0.7)
    ax.plot(wav, mean_spec, color="#E45756", linewidth=2.2, label="Mean")
    ax.fill_between(wav, p16, p84, color="#E45756", alpha=0.18, label="16th-84th percentile")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Transmission")
    ax.set_title("Lyman-alpha transmission spectra at z=7.4985")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_transmission_overall": float(mean_spec.mean()),
        "mean_transmission_min": float(mean_spec.min()),
        "mean_transmission_max": float(mean_spec.max()),
        "wav_min": float(wav.min()),
        "wav_max": float(wav.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Anson's z=7.4985 skewer dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/for_aryana/late_end_early_start/Lya_transmission"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/for_aryana"),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    params = load_params(args.data_dir / "sl_params_z=07.4985")
    density, velocity = load_sightlines(args.data_dir / "sightlines_z=07.4985")
    transmission = load_transmission(args.data_dir / "Lya_transmission.z=07.4985")
    wav = load_wavelengths(args.data_dir / "wav_out")
    radius = np.arange(N_CELLS, dtype=np.float32) * CELL_SIZE_CMPC_H

    summary = {
        "n_skewers": int(params.shape[0]),
        "n_cells_per_skewer": N_CELLS,
        "n_wavelength_bins": N_WAV,
        "angle_stats": make_angle_plot(params, args.output_dir / "angle_distributions.png"),
        "density_stats": make_profile_plot(
            radius,
            density,
            ylabel="Density",
            title="Sightline density profiles at z=7.4985",
            output_path=args.output_dir / "density_profiles.png",
            rng=rng,
        ),
        "velocity_stats": make_profile_plot(
            radius,
            velocity,
            ylabel="Velocity",
            title="Sightline velocity profiles at z=7.4985",
            output_path=args.output_dir / "velocity_profiles.png",
            rng=rng,
        ),
        "transmission_stats": make_transmission_plot(
            wav,
            transmission,
            args.output_dir / "transmission_spectra.png",
            rng,
        ),
    }

    summary_path = args.output_dir / "summary_stats.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
