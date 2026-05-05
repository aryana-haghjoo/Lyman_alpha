#!/usr/bin/env /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3
"""
Task 4: train a 1-D CNN to predict ionized bubble size (Mpc/h) from a
mock z~7.5 Lya line (intrinsic DEIMOS template × IGM transmission).

This is more realistic than Task 1 because the input is what a spectrograph
would actually observe, not the raw transmission curve.

Split  : 80% train / 20% test
Target : log1p(bubble_size) from RT-cube multi-ray MFP
Input  : mock_spectra from results/task3_mock_lya_lines/

Multi-redshift ready: loads all mock_z*.npz files automatically.

Usage (screen session, from repo root):
    /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3 \\
        scripts/task4_train_mock_lines/train_model.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp, pearsonr, spearmanr
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOG1P_MAX = 5.71   # log1p(300 Mpc/h) — covers full MFP range at all redshifts


# ── model ──────────────────────────────────────────────────────────────────

class LyaCNN(nn.Module):
    """
    1-D CNN: mock Lya line (N_WAV,) + redshift scalar → log1p(bubble size).
    Same architecture as Task 1 for direct comparison.
    """

    def __init__(self, n_wav: int = 2000) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # 2000 → 1000
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # 1000 → 250
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            # 250 → 62
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, spec: torch.Tensor, z_norm: torch.Tensor) -> torch.Tensor:
        x = self.encoder(spec.unsqueeze(1))
        x = self.gap(x).squeeze(-1)
        x = torch.cat([x, z_norm.unsqueeze(1)], dim=1)
        return self.head(x).squeeze(-1).clamp(0.0, LOG1P_MAX)


# ── data ───────────────────────────────────────────────────────────────────

def load_all_datasets(
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(dataset_dir.glob("mock_z*.npz"))
    if not paths:
        raise FileNotFoundError(f"No mock_z*.npz files in {dataset_dir}")
    all_X, all_y, all_z = [], [], []
    for p in paths:
        d = np.load(p)
        X = d["mock_spectra"].astype(np.float32)
        y = d["bubble_size"].astype(np.float32)
        z = float(d["redshift"])
        n_templates = int(d["n_templates"])
        all_X.append(X)
        all_y.append(y)
        all_z.append(np.full(len(X), z, dtype=np.float32))
        print(f"  {p.name}: {len(X)} samples, z={z:.4f}, "
              f"{n_templates} templates, "
              f"bubble median={np.median(y):.2f} Mpc/h")
    return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_z)


# ── training / evaluation ──────────────────────────────────────────────────

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for spec, z_n, y in loader:
        spec, z_n, y = spec.to(device), z_n.to(device), y.to(device)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(spec, z_n), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(spec)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    log_preds, log_trues = [], []
    for spec, z_n, y in loader:
        log_preds.append(model(spec.to(device), z_n.to(device)).cpu().numpy())
        log_trues.append(y.numpy())
    log_preds = np.concatenate(log_preds)
    log_trues = np.concatenate(log_trues)
    mse       = float(np.mean((log_preds - log_trues) ** 2))
    y_true    = np.expm1(log_trues)
    y_pred    = np.clip(np.expm1(log_preds), 0, None)
    return mse, y_true, y_pred


def make_wandb_figures(y_true, y_pred, rmse, r2, ks):
    lim = max(float(y_true.max()), float(y_pred.max())) * 1.05 + 1e-3
    images = []

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, lim, 50)
    ax.hist(y_true, bins=bins, density=True, alpha=0.6, color="#4C78A8", label="True")
    ax.hist(y_pred, bins=bins, density=True, alpha=0.6, color="#F58518", label="Predicted")
    ax.set_xlabel("Bubble size (Mpc/h)"); ax.set_ylabel("Density")
    ax.set_title(f"Distribution  KS={ks:.3f}  RMSE={rmse:.2f}  R²={r2:.3f}")
    ax.legend(); fig.tight_layout()
    images.append(wandb.Image(fig, caption="Distribution")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=3, alpha=0.25, color="#4C78A8", rasterized=True)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True (Mpc/h)"); ax.set_ylabel("Predicted (Mpc/h)")
    ax.set_title(f"Scatter  R²={r2:.3f}"); fig.tight_layout()
    images.append(wandb.Image(fig, caption="Scatter")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred - y_true, s=3, alpha=0.25, color="#E45756", rasterized=True)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axhline( rmse, color="gray", linewidth=1, linestyle="--", label=f"+RMSE ({rmse:.2f})")
    ax.axhline(-rmse, color="gray", linewidth=1, linestyle="--", label="-RMSE")
    ax.set_xlabel("True (Mpc/h)"); ax.set_ylabel("Residual (Mpc/h)")
    ax.set_title("Residuals"); ax.legend(fontsize=8); fig.tight_layout()
    images.append(wandb.Image(fig, caption="Residuals")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for arr, label, color in [(y_true, "True", "#4C78A8"), (y_pred, "Predicted", "#F58518")]:
        s = np.sort(arr)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=color, label=label, linewidth=2)
    ax.set_xlabel("Bubble size (Mpc/h)"); ax.set_ylabel("CDF")
    ax.set_title(f"CDF  KS={ks:.3f}"); ax.legend(); fig.tight_layout()
    images.append(wandb.Image(fig, caption="CDF")); plt.close(fig)

    return images


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir",    type=Path,
                   default=PROJECT_ROOT / "results/task3_mock_lya_lines")
    p.add_argument("--output-dir",     type=Path,
                   default=PROJECT_ROOT / "results/task4_train_mock_lines")
    p.add_argument("--epochs",         type=int,   default=300)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--test-frac",      type=float, default=0.2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--wandb-project",  type=str,
                   default="lyman_alpha_task4_mock_lines")
    p.add_argument("--no-wandb",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets …")
    X_raw, y_raw, z_raw = load_all_datasets(args.dataset_dir)
    n_redshifts = len(np.unique(z_raw))
    print(f"Total: {len(X_raw)} samples across {n_redshifts} redshift(s)")

    N     = len(X_raw)
    y_log = np.log1p(y_raw)
    idx   = rng.permutation(N)
    X_raw, y_log, y_raw, z_raw = X_raw[idx], y_log[idx], y_raw[idx], z_raw[idx]

    n_test  = int(N * args.test_frac)
    n_train = N - n_test
    tr_sl, te_sl = slice(0, n_train), slice(n_train, None)
    print(f"Split: {n_train} train / {n_test} test")

    mu    = X_raw[tr_sl].mean(axis=0, keepdims=True)
    sigma = X_raw[tr_sl].std(axis=0,  keepdims=True) + 1e-8
    X_norm = (X_raw - mu) / sigma

    z_mu  = z_raw[tr_sl].mean()
    z_sig = z_raw[tr_sl].std() + 1e-8
    z_norm = (z_raw - z_mu) / z_sig

    np.savez(args.output_dir / "normalisation_stats.npz",
             spec_mu=mu, spec_sigma=sigma,
             z_mu=np.float32(z_mu), z_sigma=np.float32(z_sig))

    def make_loader(sl, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X_norm[sl]),
                           torch.from_numpy(z_norm[sl]),
                           torch.from_numpy(y_log[sl]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(tr_sl, shuffle=True)
    test_loader  = make_loader(te_sl)

    model    = LyaCNN(n_wav=X_raw.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=dict(epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, weight_decay=args.weight_decay,
                        n_train=n_train, n_test=n_test,
                        n_redshifts=n_redshifts, n_params=n_params,
                        device=str(device)),
        )

    best_test_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device)
        test_mse, y_true, y_pred = eval_epoch(model, test_loader, device)
        sched.step()

        rmse  = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r2    = float(1 - np.var(y_pred - y_true) / np.var(y_true))
        rho_p = float(pearsonr(y_true, y_pred)[0])
        rho_s = float(spearmanr(y_true, y_pred)[0])
        ks, _ = ks_2samp(y_true, y_pred)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  train={train_loss:.5f}  "
                  f"test_mse={test_mse:.5f}  RMSE={rmse:.3f}  R²={r2:.3f}")

        if use_wandb:
            figs = make_wandb_figures(y_true, y_pred, rmse, r2, float(ks))
            wandb.log({"epoch": epoch, "train/loss": train_loss,
                       "test/mse_log1p": test_mse, "test/rmse_mpc_h": rmse,
                       "test/r2": r2, "test/pearson_r": rho_p,
                       "test/spearman_rho": rho_s, "test/ks_stat": float(ks),
                       "lr": sched.get_last_lr()[0],
                       "eval/distribution": figs[0], "eval/scatter": figs[1],
                       "eval/residuals": figs[2], "eval/cdf": figs[3]})

    print(f"\nBest test MSE (log1p): {best_test_mse:.5f}")

    model.load_state_dict(torch.load(args.output_dir / "best_model.pt",
                                     map_location=device))
    _, y_true, y_pred = eval_epoch(model, test_loader, device)
    rmse  = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    r2    = float(1 - np.var(y_pred - y_true) / np.var(y_true))
    ks, _ = ks_2samp(y_true, y_pred)
    print(f"Final (best model)  RMSE={rmse:.4f} Mpc/h  R²={r2:.4f}  KS={ks:.4f}")

    if use_wandb:
        wandb.log({"final/rmse_mpc_h": rmse, "final/r2": r2, "final/ks": float(ks)})
        wandb.finish()

    np.savez(args.output_dir / "test_predictions.npz",
             y_true=y_true, y_pred=y_pred, redshift=z_raw[te_sl])
    print(f"Saved → {args.output_dir / 'test_predictions.npz'}")


if __name__ == "__main__":
    main()
