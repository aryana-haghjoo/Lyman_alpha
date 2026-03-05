# Training Pipeline (Bubble Size Prediction)

This directory trains a model to predict bubble size (`target_peak_radius_mpc_h`) from RT snapshot-derived features.

## 1) Install training dependencies

```bash
python -m pip install numpy pandas scikit-learn pyyaml joblib wandb tools21cm matplotlib
```

## 2) Build dataset from snapshots

```bash
cd /home/aryana/Documents/GitHub/Lyman_alpha
PYTHONPATH=src python3 training/build_dataset_tools21cm.py \
  --data-dir data/for_aryana/late_end_early_start \
  --output-csv training/data/bubble_size_dataset.csv \
  --threshold 0.5 \
  --iterations 200000
```

## 3) Train with 80/20 split

`training/config.yaml` already uses `val_fraction: 0.2`.

```bash
cd /home/aryana/Documents/GitHub/Lyman_alpha
python3 training/train_bubble_size.py --config training/config.yaml
```

## 4) Weights & Biases (W&B)

Login:

```bash
wandb login
```

Single run:

```bash
python3 training/train_bubble_size.py --config training/config.yaml
```

Training writes:
- `training/artifacts/metrics.json`
- `training/artifacts/validation_report.json`
- `training/artifacts/val_predictions.csv`
- `training/artifacts/val_pred_vs_true.png`
- `training/artifacts/val_residual_hist.png`
- `training/artifacts/val_residual_vs_true.png`

Key quantitative validation metrics include:
- `val_rmse`, `val_mae`, `val_r2`
- `val_explained_variance`
- `val_median_ae`, `val_max_ae`
- `val_bias_mean_error`
- `val_mape_percent`, `val_wape_percent`
- `val_pearson_r`, `val_fit_slope`, `val_fit_intercept`

Create a sweep:

```bash
wandb sweep training/sweep.yaml
```

Then run agent (replace SWEEP_ID with printed value, like `user/project/abc123`):

```bash
wandb agent SWEEP_ID
```
