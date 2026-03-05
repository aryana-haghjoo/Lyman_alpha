from __future__ import annotations

import math

import numpy as np


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=(n, 3))
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / norm


def sample_mfp_distances(
    ionized_mask: np.ndarray,
    n_rays: int = 5000,
    box_size_mpc_h: float = 200.0,
    step_fraction: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """
    Estimate bubble size distribution by the MFP skewer method.

    For each ray:
    - choose a random ionized starting voxel
    - cast in a random isotropic direction
    - measure distance to first neutral-cell encounter
    """
    if ionized_mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {ionized_mask.shape}")
    if not np.any(ionized_mask):
        raise ValueError("Mask has no ionized cells at the chosen threshold.")

    n_grid = ionized_mask.shape[0]
    if ionized_mask.shape != (n_grid, n_grid, n_grid):
        raise ValueError(f"Mask must be cubic. Got shape: {ionized_mask.shape}")

    cell = box_size_mpc_h / n_grid
    step = step_fraction * cell
    max_steps = int(math.ceil(2.0 * box_size_mpc_h / step))

    rng = np.random.default_rng(seed)
    starts = np.argwhere(ionized_mask)
    directions = random_unit_vectors(n_rays, rng)
    pick = rng.integers(0, len(starts), size=n_rays)

    out = np.empty(n_rays, dtype=np.float32)

    for i in range(n_rays):
        start_idx = starts[pick[i]]
        pos = (start_idx.astype(np.float64) + 0.5) * cell
        d = directions[i]

        traveled = 0.0
        found_edge = False
        for _ in range(max_steps):
            pos = (pos + d * step) % box_size_mpc_h
            traveled += step

            voxel = np.floor(pos / cell).astype(int)
            if not ionized_mask[voxel[0], voxel[1], voxel[2]]:
                found_edge = True
                break

        out[i] = traveled if found_edge else np.nan

    return out[np.isfinite(out)]


def histogram(distances: np.ndarray, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(distances, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts

