#!/usr/bin/env /home/aryana/Documents/GitHub/Lyman_alpha/ly_a/bin/python3
"""Generate all figures for the Anson report."""
from pathlib import Path
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ks_2samp
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import interp1d

ROOT    = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

JADES_DIR = Path("/home/aryana/Documents/GitHub/multimodal_superresolution/data/JADES/DR4")
LYA_REST  = 1216.0
UM_TO_AA  = 1e4

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
})

# ── load simulation data ──────────────────────────────────────────────────
d1   = np.load(ROOT / "results/task1_transmission_to_bubble_size/dataset_z7.4985.npz")
wav  = d1["wav"]
T    = d1["transmission"]
mfp  = d1["bubble_size"]

tp1  = np.load(ROOT / "results/task1_transmission_to_bubble_size/test_predictions.npz")
tp4  = np.load(ROOT / "results/task4_train_mock_lines/test_predictions.npz")
t6   = np.load(ROOT / "results/task6_evaluate_jades/jades_predictions.npz")
mock = np.load(ROOT / "results/task3_mock_lya_lines/mock_z7.4985.npz")["mock_spectra"]

print("Data loaded.")

# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Example transmission spectra coloured by bubble size
# ════════════════════════════════════════════════════════════════════════════
print("Figure 1: example transmission spectra …")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

idx = np.argsort(mfp)
picks = idx[np.linspace(100, len(idx)-100, 8, dtype=int)]
cmap = plt.cm.plasma
colors = cmap(np.linspace(0.1, 0.9, len(picks)))

ax = axes[0]
for i, col in zip(picks, colors):
    ax.plot(wav, T[i], color=col, lw=0.9, alpha=0.85)
sm = plt.cm.ScalarMappable(cmap=cmap,
     norm=plt.Normalize(mfp[picks].min(), mfp[picks].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="RT-cube MFP (Mpc/h)")
ax.axvline(LYA_REST, color="red", lw=0.8, ls="--", alpha=0.6)
ax.set_xlim(1170, 1320); ax.set_ylim(-0.05, 1.1)
ax.set_xlabel("Rest wavelength (Å)"); ax.set_ylabel("Transmission $T(\\lambda)$")
ax.set_title("(a) Simulation transmission spectra")

ax = axes[1]
ax.hist(mfp, bins=40, color="#4C78A8", edgecolor="white", lw=0.3)
ax.set_xlabel("RT-cube MFP (Mpc/h)"); ax.set_ylabel("Count")
ax.set_title(f"(b) RT-cube MFP distribution (z=7.4985)\nMedian={np.median(mfp):.1f} Mpc/h")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_data_overview.pdf", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — The circular (old) approach: show target IS derived from input
# ════════════════════════════════════════════════════════════════════════════
print("Figure 2: circular approach …")
# Recompute transmission-edge bubble size
C_CONV = 3e5 / (LYA_REST * 1357.0)
red_mask = wav >= LYA_REST
wav_red  = wav[red_mask]
T_red    = T[:, red_mask]
above    = T_red >= 0.5
has_edge = above.any(axis=1)
first_idx = np.argmax(above, axis=1)
lam_edge  = np.where(has_edge, wav_red[first_idx], LYA_REST)
R_edge    = np.maximum(lam_edge - LYA_REST, 0.0) * C_CONV  # Mpc/h

# Show high correlation between T at 1218 Å and R_edge
i1218 = np.argmin(np.abs(wav - 1218.0))
r_circ, _ = pearsonr(T[:, i1218], R_edge)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.scatter(T[:, i1218], R_edge, s=2, alpha=0.15, color="#4C78A8", rasterized=True)
ax.set_xlabel(f"Transmission at {wav[i1218]:.1f} Å (the input)")
ax.set_ylabel("Transmission-edge $R$ (the target)")
ax.set_title(f"(a) Target vs input at one wavelength\nPearson $r = {r_circ:.3f}$ — target is\n"
             "almost entirely determined by the input")

ax = axes[1]
# Illustrate: T(lambda) → find T=0.5 crossing → R
wav_ex = np.linspace(1200, 1280, 500)
# Three example "T" profiles with different edges
for r_ex, col, lab in [(0.05,"#E45756","R=0.05 Mpc/h"), (0.15,"#72B7B2","R=0.15 Mpc/h"), (0.30,"#54A24B","R=0.30 Mpc/h")]:
    edge = LYA_REST + r_ex / C_CONV
    T_ex = np.where(wav_ex < LYA_REST, 0.0, np.clip((wav_ex - LYA_REST)/(edge - LYA_REST + 1e-3), 0, 1)**0.5)
    T_ex = np.where(wav_ex > edge, 1.0, T_ex)
    ax.plot(wav_ex, T_ex, color=col, lw=1.8, label=lab)
    ax.axvline(edge, color=col, lw=0.8, ls=":", alpha=0.8)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="$T=0.5$ threshold")
ax.axvline(LYA_REST, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Rest wavelength (Å)"); ax.set_ylabel("Transmission $T(\\lambda)$")
ax.set_title("(b) How $R_{\\rm edge}$ is read off $T(\\lambda)$\n"
             "— target is a trivial feature of the input")
ax.legend(loc="lower right"); ax.set_xlim(1195, 1270)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_circular_approach.pdf", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Pearson r per wavelength bin: transmission vs RT-cube MFP
# ════════════════════════════════════════════════════════════════════════════
print("Figure 3: Pearson r analysis …")
r_vals = np.array([pearsonr(T[:, i], mfp)[0] for i in range(0, len(wav), 5)])
wav_sub = wav[::5]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(wav_sub, r_vals, color="#4C78A8", lw=1.2)
ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.axhline(0.3, color="green", lw=0.8, ls=":", alpha=0.7, label="$r=0.3$ (learnable threshold)")
ax.fill_between(wav_sub, r_vals, 0,
                where=np.abs(r_vals) > 0, alpha=0.15, color="#4C78A8")
ax.axvline(LYA_REST, color="red", lw=0.8, ls="--", alpha=0.6, label="Ly$\\alpha$ 1216 Å")
ax.set_xlabel("Rest wavelength (Å)")
ax.set_ylabel("Pearson $r$ (transmission vs RT-cube MFP)")
ax.set_title("Correlation between transmission and RT-cube MFP at each wavelength\n"
             f"Max $|r|$ = {np.max(np.abs(r_vals)):.3f} — no wavelength bin carries learnable signal")
ax.set_xlim(wav.min(), wav.max())
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_pearson_r.pdf", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Task 1 & Task 4 results (scatter + CDF)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 4: Task 1 & 4 results …")
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for row_i, (tp, title) in enumerate([(tp1, "Task 1: raw transmission input"),
                                      (tp4, "Task 4: mock Lya line input")]):
    yt, yp = tp["y_true"], tp["y_pred"]
    r2  = float(1 - np.var(yp - yt) / np.var(yt))
    rp, _ = pearsonr(yt, yp)
    ks, _ = ks_2samp(yt, yp)
    lim = max(yt.max(), yp.max()) * 1.05

    # Scatter
    ax = axes[row_i, 0]
    ax.scatter(yt, yp, s=2, alpha=0.2, color="#4C78A8", rasterized=True)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True MFP (Mpc/h)"); ax.set_ylabel("Predicted MFP (Mpc/h)")
    ax.set_title(f"({chr(97+row_i*2)}) {title}\n$R^2={r2:.3f}$, Pearson $r={rp:.3f}$")

    # CDF
    ax = axes[row_i, 1]
    for arr, label, color in [(yt, "True", "#4C78A8"), (yp, "Predicted", "#F58518")]:
        s = np.sort(arr)
        ax.plot(s, np.arange(1, len(s)+1)/len(s), color=color, lw=2, label=label)
    ax.set_xlabel("Bubble size (Mpc/h)"); ax.set_ylabel("CDF")
    ax.set_title(f"({chr(98+row_i*2)}) CDF comparison  KS={ks:.3f}")
    ax.legend()

fig.suptitle("Model performance: Task 1 (transmission) vs Task 4 (mock Lya lines)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_model_results.pdf", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — JADES templates: intrinsic vs attenuated
# ════════════════════════════════════════════════════════════════════════════
print("Figure 5: JADES intrinsic vs attenuated …")
cat = Table.read(JADES_DIR / "Combined_DR4_external_v1.2.1.fits")
z_jades = np.array(cat["z_Spec"], dtype=float)
flag    = np.array(cat["z_Spec_flag"], dtype=str)

all_prism  = glob.glob(str(JADES_DIR / "**" / "*prism*x1d*.fits"), recursive=True)
id_to_path = {int(Path(p).stem.split("_")[4].split("-")[-1]): p for p in all_prism}

def load_jades(obj_id):
    path = id_to_path.get(obj_id)
    if not path: return None, None
    with fits.open(path) as h:
        data = h["EXTRACT5PIX1D"].data
        w = np.array(data["WAVELENGTH"], dtype=float) * UM_TO_AA
        f = np.array(data["FLUX"], dtype=float)
        e = np.array(data["FLUX_ERR"], dtype=float)
    f[(e <= 0) | ~np.isfinite(f)] = np.nan
    return w, f

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: intrinsic z=4-5.5
ax = axes[0]
tmpl = cat[(z_jades >= 4.0) & (z_jades <= 5.5) & np.isin(flag, ["A","B"])]
colors = plt.cm.Blues(np.linspace(0.35, 0.9, 8))
n = 0
for row, col in zip(tmpl, colors):
    if n >= 8: break
    w, f = load_jades(int(row["NIRSpec_ID"]))
    if w is None: continue
    z_o = float(row["z_Spec"])
    wr = w / (1 + z_o)
    m  = np.abs(wr - LYA_REST) < 80
    wr, fr = wr[m], f[m]
    if len(wr) < 5 or np.nanmax(fr) <= 0: continue
    ax.plot(wr, fr / np.nanmax(fr), color=col, lw=1.0, alpha=0.75)
    n += 1
ax.axvline(LYA_REST, color="red", lw=0.8, ls="--", label="Ly$\\alpha$ 1216 Å")
ax.axhline(0, color="gray", lw=0.4)
ax.set_xlim(1140, 1310); ax.set_ylim(-0.25, 1.4)
ax.set_xlabel("Rest wavelength (Å)"); ax.set_ylabel("Normalised flux")
ax.set_title("(a) Intrinsic Lya templates  (JADES, z=4–5.5)\n"
             "IGM transparent: observed $\\approx$ emitted")
ax.legend()

# Right: attenuated z=7-8
ax = axes[1]
evl = cat[(z_jades >= 7.0) & (z_jades < 8.0)]
colors2 = plt.cm.Oranges(np.linspace(0.35, 0.9, 8))
n = 0
for row, col in zip(evl, colors2):
    if n >= 8: break
    w, f = load_jades(int(row["NIRSpec_ID"]))
    if w is None: continue
    z_o = float(row["z_Spec"])
    wr = w / (1 + z_o)
    m  = np.abs(wr - LYA_REST) < 80
    wr, fr = wr[m], f[m]
    if len(wr) < 5 or np.nanmax(fr) <= 0: continue
    ax.plot(wr, fr / np.nanmax(fr), color=col, lw=1.0, alpha=0.75)
    n += 1
ax.axvline(LYA_REST, color="red", lw=0.8, ls="--", label="Ly$\\alpha$ 1216 Å")
ax.axhline(0, color="gray", lw=0.4)
ax.set_xlim(1140, 1310); ax.set_ylim(-0.25, 1.4)
ax.set_xlabel("Rest wavelength (Å)")
ax.set_title("(b) Evaluation spectra  (JADES, z=7–8)\n"
             "Reionization era: GP trough + damping wing")
ax.legend()

fig.suptitle("JWST/NIRSpec prism — same instrument for templates and evaluation", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_jades_spectra.pdf", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — Mock Lya line: intrinsic × T = mock
# ════════════════════════════════════════════════════════════════════════════
print("Figure 6: mock Lya line examples …")
idx_sorted = np.argsort(mfp)
picks3 = idx_sorted[np.linspace(200, len(idx_sorted)-200, 3, dtype=int)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, i in zip(axes, picks3):
    Tv = T[i]
    Mv = mock[i]
    safe = Tv > 0.02
    intr = np.where(safe, Mv / Tv, np.nan)
    ylim = min(3.0, float(np.nanpercentile(intr, 97)) * 1.3)

    ax.plot(wav, Tv,   color="#54A24B", lw=1.2, label="Transmission $T$", alpha=0.8)
    ax.plot(wav, intr, color="#4C78A8", lw=1.2, label="Intrinsic template", alpha=0.7, ls="--")
    ax.plot(wav, Mv,   color="#F58518", lw=1.5, label="Mock observed", alpha=0.9)
    ax.axvline(LYA_REST, color="red", lw=0.7, ls=":", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlim(1185, 1280)
    ax.set_ylim(-0.1, max(1.5, ylim))
    ax.set_title(f"MFP = {mfp[i]:.1f} Mpc/h", fontsize=10)
    ax.set_xlabel("Rest wavelength (Å)", fontsize=9)
    ax.legend(fontsize=7.5)

axes[0].set_ylabel("Normalised flux / $T$")
fig.suptitle("Mock Lya line = Intrinsic template $\\times$ Simulation transmission", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_mock_lines.pdf", bbox_inches="tight")
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIG 7 — Task 6: recovered bubble size distribution from JADES
# ════════════════════════════════════════════════════════════════════════════
print("Figure 7: Task 6 results …")
pred_jades = t6["pred_bubble_size"]
sim_mfp    = t6["sim_bubble_size"]
z_jades_ev = t6["z_spec"]
ks6, _     = ks_2samp(pred_jades, sim_mfp)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
lim = max(np.percentile(sim_mfp, 99), np.percentile(pred_jades, 99)) * 1.1
bins = np.linspace(0, lim, 35)
ax.hist(sim_mfp,    bins=bins, density=True, alpha=0.6, color="#4C78A8",
        label=f"Simulation RT cube ($n={len(sim_mfp):,}$)")
ax.hist(pred_jades, bins=bins, density=True, alpha=0.7, color="#F58518",
        label=f"JADES z=7–8 CNN predictions ($n={len(pred_jades)}$)")
ax.set_xlabel("Bubble size (Mpc/h)"); ax.set_ylabel("Probability density")
ax.set_title(f"(a) Recovered bubble size distribution\nKS = {ks6:.3f}")
ax.legend()

ax = axes[1]
for arr, label, color in [
    (sim_mfp,    "Simulation RT cube",  "#4C78A8"),
    (pred_jades, "JADES z=7–8 (CNN)",   "#F58518"),
]:
    s = np.sort(arr)
    ax.plot(s, np.arange(1, len(s)+1)/len(s), color=color, lw=2, label=label)
ax.set_xlabel("Bubble size (Mpc/h)"); ax.set_ylabel("CDF")
ax.set_title(f"(b) Cumulative distribution  KS = {ks6:.3f}")
ax.legend()

fig.suptitle("Task 6: CNN applied to 108 real JADES z=7–8 LAEs", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_task6_results.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {FIG_DIR}")
