# Lyman_alpha
Starter pipeline for learning Lyman-alpha bubble statistics from RT simulations during reionization.

## Data
RT snapshots are expected in:

`data/for_aryana/late_end_early_start`

Each snapshot is a binary float32 file with layout:

- 1 header float
- `N x N x N x 21` values (with `N=200` for current data)

From the project email, the ionized fraction is `fion` at index `15`.

## Quick Start
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

If you prefer notebooks:

```bash
pip install jupyter matplotlib
jupyter notebook notebooks/lyman_alpha_starter.ipynb
```

Run a first pass over snapshots to get mean ionized fraction history:

```bash
PYTHONPATH=src python3 scripts/scan_snapshots.py \
  --data-dir data/for_aryana/late_end_early_start \
  --output results/ionization_history.csv
```

Run the bubble-size (MFP) estimate on one snapshot:

```bash
PYTHONPATH=src python3 scripts/run_bubble_stats.py \
  --snapshot data/for_aryana/late_end_early_start/gas_z=08.1176 \
  --n-rays 5000 \
  --threshold 0.5 \
  --output-dir results
```

## Project Layout
- `src/lyman_alpha/data.py`: binary loader + field extraction.
- `src/lyman_alpha/bubble_stats.py`: MFP skewer-based bubble size calculation.
- `scripts/scan_snapshots.py`: redshift vs mean ionized fraction table.
- `scripts/run_bubble_stats.py`: compute and save bubble-size distribution for one snapshot.
- `notebooks/lyman_alpha_starter.ipynb`: inline plots + exploratory workflow.

## Recommended First Milestones
1. Generate `ionization_history.csv` and identify representative stages (`x_HII ~ 0.2, 0.5, 0.8`).
2. Compute MFP distance distributions at those stages.
3. Validate stability vs `n_rays`, threshold choice, and random seed.
4. Convert each snapshot to ML-ready features (histogram bins + summary stats).
