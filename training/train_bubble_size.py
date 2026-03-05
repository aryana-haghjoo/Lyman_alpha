#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bubble-size regressor with 80/20 split.")
    p.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    blocked = {"snapshot", "redshift", target_column}
    return [c for c in df.columns if c not in blocked]


def apply_wandb_overrides(cfg: dict[str, Any], wb_cfg: Any) -> dict[str, Any]:
    # Supports dotted keys from sweep config, e.g. "model.max_depth".
    out = json.loads(json.dumps(cfg))
    for k, v in dict(wb_cfg).items():
        if "." not in k:
            continue
        root, leaf = k.split(".", 1)
        if root in out and isinstance(out[root], dict):
            out[root][leaf] = v
    return out


def main() -> None:
    args = parse_args()

    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split

    cfg = load_config(args.config)

    wb_enabled = bool(cfg.get("wandb", {}).get("enabled", True))
    wandb = None
    run = None
    if wb_enabled:
        import wandb

        run = wandb.init(
            project=cfg["project"],
            entity=cfg.get("entity"),
            config=cfg,
            name=cfg.get("wandb", {}).get("run_name"),
        )
        cfg = apply_wandb_overrides(cfg, run.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    output_cfg = cfg["output"]

    df = pd.read_csv(data_cfg["input_csv"])
    y = df[data_cfg["target_column"]].to_numpy(dtype=np.float32)
    feat_cols = get_feature_columns(df, data_cfg["target_column"])
    X = df[feat_cols].to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=float(data_cfg["val_fraction"]),
        random_state=int(cfg["seed"]),
        shuffle=True,
    )

    if model_cfg["type"] != "random_forest":
        raise SystemExit(f"Unsupported model type: {model_cfg['type']}")

    model = RandomForestRegressor(
        n_estimators=int(model_cfg["n_estimators"]),
        max_depth=None if model_cfg["max_depth"] is None else int(model_cfg["max_depth"]),
        min_samples_split=int(model_cfg["min_samples_split"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        random_state=int(cfg["seed"]),
        n_jobs=int(model_cfg["n_jobs"]),
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    residual_val = pred_val - y_val
    abs_err_val = np.abs(residual_val)
    denom = np.maximum(np.abs(y_val), 1e-8)
    mape_val = float(np.mean(abs_err_val / denom) * 100.0)
    wape_val = float(np.sum(abs_err_val) / np.maximum(np.sum(np.abs(y_val)), 1e-8) * 100.0)
    corr_val = float(np.corrcoef(y_val, pred_val)[0, 1]) if len(y_val) > 1 else float("nan")
    slope, intercept = np.polyfit(y_val, pred_val, 1)
    slope = float(slope)
    intercept = float(intercept)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "train_r2": float(r2_score(y_train, pred_train)),
        "val_mae": float(mean_absolute_error(y_val, pred_val)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
        "val_r2": float(r2_score(y_val, pred_val)),
        "val_explained_variance": float(explained_variance_score(y_val, pred_val)),
        "val_median_ae": float(median_absolute_error(y_val, pred_val)),
        "val_max_ae": float(np.max(abs_err_val)),
        "val_bias_mean_error": float(np.mean(residual_val)),
        "val_mape_percent": mape_val,
        "val_wape_percent": wape_val,
        "val_pearson_r": corr_val,
        "val_fit_slope": slope,
        "val_fit_intercept": intercept,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(X.shape[1]),
    }

    out_dir = Path(output_cfg["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / output_cfg["model_name"]
    metrics_path = out_dir / "metrics.json"
    val_report_path = out_dir / "validation_report.json"
    features_path = out_dir / "feature_columns.json"
    val_pred_path = out_dir / "val_predictions.csv"
    plot_pred_true = out_dir / "val_pred_vs_true.png"
    plot_resid_hist = out_dir / "val_residual_hist.png"
    plot_resid_true = out_dir / "val_residual_vs_true.png"

    # Save raw validation predictions for later analysis.
    pd.DataFrame(
        {
            "y_true": y_val,
            "y_pred": pred_val,
            "residual": residual_val,
            "abs_error": abs_err_val,
            "ape_percent": abs_err_val / np.maximum(np.abs(y_val), 1e-8) * 100.0,
        }
    ).to_csv(val_pred_path, index=False)

    # Plot 1: predicted vs true.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_val, pred_val, alpha=0.8, s=28)
    mn = float(min(np.min(y_val), np.min(pred_val)))
    mx = float(max(np.max(y_val), np.max(pred_val)))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, label="ideal")
    ax.set_xlabel("True bubble size [cMpc/h]")
    ax.set_ylabel("Predicted bubble size [cMpc/h]")
    ax.set_title("Validation: Predicted vs True")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_pred_true, dpi=160)
    plt.close(fig)

    # Plot 2: residual histogram.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(residual_val, bins=20, alpha=0.85)
    ax.axvline(0.0, color="k", linestyle="--", lw=1.5)
    ax.set_xlabel("Residual (pred - true) [cMpc/h]")
    ax.set_ylabel("Count")
    ax.set_title("Validation Residual Distribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_resid_hist, dpi=160)
    plt.close(fig)

    # Plot 3: residual vs true.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(y_val, residual_val, alpha=0.8, s=28)
    ax.axhline(0.0, color="k", linestyle="--", lw=1.5)
    ax.set_xlabel("True bubble size [cMpc/h]")
    ax.set_ylabel("Residual (pred - true) [cMpc/h]")
    ax.set_title("Validation Residuals vs True")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_resid_true, dpi=160)
    plt.close(fig)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    val_report = {
        "target_column": data_cfg["target_column"],
        "val_size": int(len(y_val)),
        "y_true_mean": float(np.mean(y_val)),
        "y_true_std": float(np.std(y_val)),
        "y_pred_mean": float(np.mean(pred_val)),
        "y_pred_std": float(np.std(pred_val)),
        "metrics": metrics,
    }
    val_report_path.write_text(json.dumps(val_report, indent=2))
    features_path.write_text(json.dumps(feat_cols, indent=2))

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved: {model_path}")
    print(f"Validation report: {val_report_path}")
    print(f"Validation predictions: {val_pred_path}")
    print(f"Plot saved: {plot_pred_true}")
    print(f"Plot saved: {plot_resid_hist}")
    print(f"Plot saved: {plot_resid_true}")

    if run is not None:
        run.log(metrics)
        run.log(
            {
                "val_pred_vs_true": wandb.Image(str(plot_pred_true)),
                "val_residual_hist": wandb.Image(str(plot_resid_hist)),
                "val_residual_vs_true": wandb.Image(str(plot_resid_true)),
            }
        )
        run.summary["model_path"] = str(model_path)
        run.summary["metrics_path"] = str(metrics_path)
        run.summary["validation_report_path"] = str(val_report_path)
        run.finish()


if __name__ == "__main__":
    main()
