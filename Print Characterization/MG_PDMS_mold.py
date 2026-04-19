"""
Print Accuracy Analysis — Chip5 Positive Mold
Chip5 MG2:1  (double-molded, 2:1 scale)
Chip5 MG1:1  (double-molded, 1:1 scale)
Saves all figures to the same directory as this script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS_W   = "#378ADD"
COLORS_H   = "#1D9E75"
RED        = "#E24B4A"
AMBER      = "#EF9F27"
GRAY_LIGHT = "#D3D1C7"
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRAY_LIGHT,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
})

OUT = Path(__file__).parent

# ── Data ──────────────────────────────────────────────────────────────────────
datasets = {
    "Chip5 MG2:1": {
        "Hemi-ellipsoid": {
            "true_w": 30, "true_h": 30,
            "width":  [29.252, 27.95, 27.463, 28.113, 28.6],
            "height": [34.125, 34.938, 33.638, 33.8, 33.962],
        },
        "Cone": {
            "true_w": 15, "true_h": 30,
            "width":  [15.925, 12.513, 13.0, 14.625, 13.975],
            "height": [28.762, 37.7, 31.525, 29.25, 32.175],
        },
        "Hyperbolic": {
            "true_w": 15, "true_h": 30,
            "width":  [8.45, 8.613, 8.125, 8.613, 13.0],
            "height": [26.195, 28.6, 28.275, 24.7, 22.75],
        },
        "Tri. Prism": {
            "true_w": 15, "true_h": 30,
            "width":  [13.375, 11.375, 11.537, 12.351, 12.35],
            "height": [31.2, 33.312, 32.825, 34.613, 34.938],
        },
        "Sq. Pyramid": {
            "true_w": 15, "true_h": 30,
            "width":  [13.0, 13.488, 13.325, 14.3, 14.3],
            "height": [28.925, 30.387, 30.387, 32.988, 33.962],
        },
    },
    "Chip5 MG1:1": {
        "Hemi-ellipsoid": {
            "true_w": 15, "true_h": 15,
            "width":  [13.65, 14.3, 15.113, 14.3, 13.975],
            "height": [18.2, 20.312, 20.8, 19.663, 19.175],
        },
        "Cone": {
            "true_w": 15, "true_h": 15,
            "width":  [12.514, 14.787, 12.513, 15.6, 14.787],
            "height": [19.175, 20.15, 20.8, 20.8, 16.575],
        },
        "Hyperbolic": {
            "true_w": 15, "true_h": 15,
            "width":  [10.4, 11.7, 13.162, 11.537, 10.4],
            "height": [18.038, 16.738, 18.85, 18.525, 16.738],
        },
        "Tri. Prism": {
            "true_w": 15, "true_h": 15,
            "width":  [11.537, 14.138, 12.35, 15.113, 11.05],
            "height": [17.713, 20.312, 21.288, 22.263, 21.94],
        },
        "Sq. Pyramid": {
            "true_w": 15, "true_h": 15,
            "width":  [12.675, 13.162, 14.95, 14.625, 13.0],
            "height": [17.713, 20.637, 19.5, 19.998, 21.288],
        },
    },
}

# ── Stats helper ──────────────────────────────────────────────────────────────
def compute_stats(values, true_val):
    arr      = np.array(values, dtype=float)
    mean     = arr.mean()
    sd       = arr.std(ddof=1)
    bias     = mean - true_val
    pct_errs = np.abs(arr - true_val) / true_val * 100
    avg_pct  = pct_errs.mean()
    return dict(mean=mean, sd=sd, bias=bias, avg_pct=avg_pct,
                pct_errs=pct_errs, values=arr, true=true_val)

all_results = {}
for ds_name, geos in datasets.items():
    all_results[ds_name] = {}
    for gname, data in geos.items():
        all_results[ds_name][gname] = {
            "width":  compute_stats(data["width"],  data["true_w"]),
            "height": compute_stats(data["height"], data["true_h"]),
        }

DS_NAMES  = list(datasets.keys())
GEO_NAMES = list(list(datasets.values())[0].keys())
x = np.arange(len(GEO_NAMES))
W = 0.35

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Avg % error, side-by-side datasets, width & height
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, ds in zip(axes, DS_NAMES):
    res = all_results[ds]
    avg_w = [res[g]["width"]["avg_pct"]  for g in GEO_NAMES]
    avg_h = [res[g]["height"]["avg_pct"] for g in GEO_NAMES]
    ax.bar(x - W/2, avg_w, W, label="Width",  color=COLORS_W)
    ax.bar(x + W/2, avg_h, W, label="Height", color=COLORS_H)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title(f"{ds} — avg % error")
    ax.set_ylabel("Average % error")
    ax.legend()
fig.suptitle("Chip5 Positive Mold — Average percent error vs. true", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "chip5_fig1_avg_pct_error.png")
plt.close(fig)
print("Saved chip5_fig1_avg_pct_error.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Bias side-by-side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ds in zip(axes, DS_NAMES):
    res = all_results[ds]
    bias_w = [res[g]["width"]["bias"]  for g in GEO_NAMES]
    bias_h = [res[g]["height"]["bias"] for g in GEO_NAMES]
    ax.bar(x - W/2, bias_w, W, color=[COLORS_W if v >= 0 else RED for v in bias_w])
    ax.bar(x + W/2, bias_h, W, color=[COLORS_H if v >= 0 else RED for v in bias_h])
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Bias (mm)  [+ = over, − = under]")
    ax.set_title(f"{ds} — dimensional bias")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color=COLORS_W, label="Width over"),
    plt.Rectangle((0,0),1,1, color=COLORS_H, label="Height over"),
    plt.Rectangle((0,0),1,1, color=RED,      label="Under-sized"),
])
fig.suptitle("Chip5 Positive Mold — Dimensional bias (mean − true)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "chip5_fig2_bias.png")
plt.close(fig)
print("Saved chip5_fig2_bias.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Std dev side-by-side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, ds in zip(axes, DS_NAMES):
    res = all_results[ds]
    sd_w = [res[g]["width"]["sd"]  for g in GEO_NAMES]
    sd_h = [res[g]["height"]["sd"] for g in GEO_NAMES]
    ax.bar(x - W/2, sd_w, W, label="Width σ",  color=COLORS_W)
    ax.bar(x + W/2, sd_h, W, label="Height σ", color=COLORS_H)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Std dev σ (mm)")
    ax.set_title(f"{ds} — std deviation")
    ax.legend()
fig.suptitle("Chip5 Positive Mold — Sample standard deviation", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "chip5_fig3_std_dev.png")
plt.close(fig)
print("Saved chip5_fig3_std_dev.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Per-sample scatter, both datasets (2 rows × 5 cols)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for row, ds in enumerate(DS_NAMES):
    res = all_results[ds]
    for col, gname in enumerate(GEO_NAMES):
        ax = axes[row][col]
        wv = res[gname]["width"]["values"]
        hv = res[gname]["height"]["values"]
        tw = res[gname]["width"]["true"]
        th = res[gname]["height"]["true"]
        xs = np.arange(1, len(wv)+1)
        ax.scatter(xs, wv, color=COLORS_W, s=40, zorder=3, label="Width")
        ax.scatter(xs, hv, color=COLORS_H, s=40, zorder=3, label="Height")
        ax.axhline(tw, color=COLORS_W, linewidth=1.0, linestyle="--")
        ax.axhline(th, color=COLORS_H, linewidth=1.0, linestyle="--")
        ax.axhline(res[gname]["width"]["mean"],  color=COLORS_W, linewidth=0.8, linestyle=":")
        ax.axhline(res[gname]["height"]["mean"], color=COLORS_H, linewidth=0.8, linestyle=":")
        ax.set_title(f"{gname}\n{ds}", fontsize=8)
        ax.set_xlabel("Sample", fontsize=8)
        ax.set_ylabel("mm", fontsize=8)
        ax.set_xticks(xs)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
fig.suptitle("Chip5 Per-sample measurements — dashed = true, dotted = mean", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "chip5_fig4_per_sample_scatter.png")
plt.close(fig)
print("Saved chip5_fig4_per_sample_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Per-sample % error bars, both datasets (2 rows × 5 cols)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for row, ds in enumerate(DS_NAMES):
    res = all_results[ds]
    for col, gname in enumerate(GEO_NAMES):
        ax = axes[row][col]
        pw = res[gname]["width"]["pct_errs"]
        ph = res[gname]["height"]["pct_errs"]
        xs = np.arange(1, len(pw)+1)
        bw = 0.35
        ax.bar(xs - bw/2, pw, bw,
               color=[RED if v > 3 else AMBER if v > 1 else COLORS_W for v in pw])
        ax.bar(xs + bw/2, ph, bw,
               color=[RED if v > 3 else AMBER if v > 1 else COLORS_H for v in ph])
        ax.set_title(f"{gname}\n{ds}", fontsize=8)
        ax.set_xlabel("Sample", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=7)
        if col == 0:
            ax.set_ylabel("% error", fontsize=8)
fig.suptitle("Chip5 Per-sample % error  |  Blue/green < 1%  ·  Amber 1–3%  ·  Red > 3%", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "chip5_fig5_pct_error_bars.png")
plt.close(fig)
print("Saved chip5_fig5_pct_error_bars.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Width bias comparison: Master Mold vs Chip5 (MG1:1 only, true=15)
# Shows how the positive mold propagated or corrected error
# ─────────────────────────────────────────────────────────────────────────────
master_mg1_bias_w = {
    "Hemi-ellipsoid": -0.4409,
    "Cone":           -1.0158,
    "Hyperbolic":     -2.5220,
    "Tri. Prism":     -2.0918,
    "Sq. Pyramid":    -1.1591,
}
master_mg1_bias_h = {
    "Hemi-ellipsoid": +2.1879,
    "Cone":           -1.6190,
    "Hyperbolic":     -3.9019,
    "Tri. Prism":     +0.3729,
    "Sq. Pyramid":    +0.9597,
}
chip5_mg11_res = all_results["Chip5 MG1:1"]
chip5_bias_w = [chip5_mg11_res[g]["width"]["bias"]  for g in GEO_NAMES]
chip5_bias_h = [chip5_mg11_res[g]["height"]["bias"] for g in GEO_NAMES]
master_bw    = [master_mg1_bias_w[g] for g in GEO_NAMES]
master_bh    = [master_mg1_bias_h[g] for g in GEO_NAMES]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bw2 = 0.35
for ax, (mb, cb, dim, col_m, col_c, label) in zip(axes, [
    (master_bw, chip5_bias_w, "Width",  "#85B7EB", COLORS_W, "Width"),
    (master_bh, chip5_bias_h, "Height", "#9FE1CB", COLORS_H, "Height"),
]):
    ax.bar(x - bw2/2, mb, bw2, label="Master MG1:1", color=col_m)
    ax.bar(x + bw2/2, cb, bw2, label="Chip5 MG1:1",  color=col_c)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Bias (mm)")
    ax.set_title(f"{dim} bias: Master vs Chip5 (MG1:1)")
    ax.legend()
fig.suptitle("Bias propagation — Master MG1:1 → Chip5 MG1:1 (positive mold)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "chip5_fig6_bias_propagation_MG1.png")
plt.close(fig)
print("Saved chip5_fig6_bias_propagation_MG1.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary tables
# ─────────────────────────────────────────────────────────────────────────────
for ds_name, res in all_results.items():
    print(f"\n{'='*80}")
    print(f"  {ds_name}")
    print(f"{'='*80}")
    print(f"{'Geometry':<18} {'Dim':<8} {'True':>6} {'Mean':>8} {'Std dev':>8} {'Bias':>8} {'Avg %err':>9}")
    print("-"*80)
    for gname, dims in res.items():
        for dim in ("width", "height"):
            s = dims[dim]
            print(f"{gname if dim=='width' else '':<18} {dim:<8} "
                  f"{s['true']:>6.0f} {s['mean']:>8.3f} {s['sd']:>8.4f} "
                  f"{s['bias']:>+8.4f} {s['avg_pct']:>8.2f}%")
        print("-"*80)

print(f"\nAll figures saved to: {OUT.resolve()}/")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Bias propagation: Master MG2:1 → Chip5 MG2:1
# Master MG2:1 bias values from previously computed results:
#   HemiEllipsoid: true_w=30, true_h=30  |  all others: true_w=15, true_h=30
# ─────────────────────────────────────────────────────────────────────────────
master_mg21_bias_w = {
    "Hemi-ellipsoid": -0.1064,
    "Cone":           +0.0600,
    "Hyperbolic":     -1.8766,
    "Tri. Prism":     +1.4224,
    "Sq. Pyramid":    +0.2113,
}
master_mg21_bias_h = {
    "Hemi-ellipsoid": -1.4863,
    "Cone":           -4.8340,
    "Hyperbolic":     -6.3532,
    "Tri. Prism":     -1.9755,
    "Sq. Pyramid":    -3.4925,
}
chip5_mg21_res = all_results["Chip5 MG2:1"]
chip5_mg21_bw = [chip5_mg21_res[g]["width"]["bias"]  for g in GEO_NAMES]
chip5_mg21_bh = [chip5_mg21_res[g]["height"]["bias"] for g in GEO_NAMES]
master_mg21_bw_list = [master_mg21_bias_w[g] for g in GEO_NAMES]
master_mg21_bh_list = [master_mg21_bias_h[g] for g in GEO_NAMES]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bw2 = 0.35
for ax, (mb, cb, dim, col_m, col_c) in zip(axes, [
    (master_mg21_bw_list, chip5_mg21_bw, "Width",  "#85B7EB", COLORS_W),
    (master_mg21_bh_list, chip5_mg21_bh, "Height", "#9FE1CB", COLORS_H),
]):
    ax.bar(x - bw2/2, mb, bw2, label="Master MG2:1", color=col_m)
    ax.bar(x + bw2/2, cb, bw2, label="Chip5 MG2:1",  color=col_c)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Bias (mm)")
    ax.set_title(f"{dim} bias: Master vs Chip5 (MG2:1)")
    ax.legend()
fig.suptitle("Bias propagation — Master MG2:1 → Chip5 MG2:1 (positive mold)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "chip5_fig7_bias_propagation_MG2.png")
plt.close(fig)
print("Saved chip5_fig7_bias_propagation_MG2.png")