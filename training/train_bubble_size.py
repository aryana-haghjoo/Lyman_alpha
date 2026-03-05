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
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "train_r2": float(r2_score(y_train, pred_train)),
        "val_mae": float(mean_absolute_error(y_val, pred_val)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
        "val_r2": float(r2_score(y_val, pred_val)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(X.shape[1]),
    }

    out_dir = Path(output_cfg["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / output_cfg["model_name"]
    metrics_path = out_dir / "metrics.json"
    features_path = out_dir / "feature_columns.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    features_path.write_text(json.dumps(feat_cols, indent=2))

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved: {model_path}")

    if run is not None:
        run.log(metrics)
        run.summary["model_path"] = str(model_path)
        run.summary["metrics_path"] = str(metrics_path)
        run.finish()


if __name__ == "__main__":
    main()
