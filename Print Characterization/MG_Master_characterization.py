"""
Master Mold Characterization — MG1:1 & MG2:1
Computes percent error, bias, mean, and std dev vs. true dimensions.
Produces side-by-side subplot figures comparing both scales.
Saves all figures to the same directory as this script.

True dimensions:
  MG1:1 — all geometries: width=15, height=15
  MG2:1 — HemiEllipsoid: width=30, height=30
           all others:    width=15, height=30
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
    "MG1:1": {
        "Hemi-ellipsoid": {
            "true_w": 15, "true_h": 15,
            "width":  [15.06, 14.343, 15.06, 14.343, 14.343, 15.06, 14.353, 14.343, 15.06, 13.626],
            "height": [15.999, 16.999, 16.238, 13.999, 19.363, 17.928, 17.549, 17.928, 18.646, 17.23],
        },
        "Cone": {
            "true_w": 15, "true_h": 15,
            "width":  [12.908, 12.191, 13.626, 13.626, 12.908, 13.626, 11.474, 13.626, 14.343, 11.514],
            "height": [12.999, 12.999, 13.999, 12.951, 13.626, 12.908, 13.626, 13.077, 13.626, 13.999],
        },
        "Hyperbolic": {
            "true_w": 15, "true_h": 15,
            "width":  [12.191, 12.191, 12.191, 12.908, 12.908, 13.626, 11.474, 12.908, 13.626, 10.757],
            "height": [10.999, 11.231, 10.999, 10.999, 10.575, 10.757, 11.474, 11.474, 11.474, 10.999],
        },
        "Tri. Prism": {
            "true_w": 15, "true_h": 15,
            "width":  [12.191, 12.908, 12.908, 12.908, 12.908, 13.626, 12.908, 13.626, 12.908, 12.191],
            "height": [15.999, 15.999, 14.999, 14.999, 15.06,  15.06,  15.777, 15.777, 15.06,  14.999],
        },
        "Sq. Pyramid": {
            "true_w": 15, "true_h": 15,
            "width":  [14.343, 13.626, 13.626, 14.343, 13.626, 14.343, 12.908, 14.343, 14.343, 12.908],
            "height": [14.999, 15.999, 15.999, 15.999, 15.777, 15.777, 16.494, 16.494, 15.06,  16.999],
        },
    },
    "MG2:1": {
        "Hemi-ellipsoid": {
            "true_w": 30, "true_h": 30,
            "width":  [31.554, 31.554, 30.837, 29.403, 29.403, 28.523, 29.061, 28.523, 29.403, 30.675],
            "height": [27.998, 28.998, 27.998, 27.878, 29.998, 27.838, 26.748, 30.998, 28.685, 27.998],
        },
        "Cone": {
            "true_w": 15, "true_h": 30,
            "width":  [15.777, 14.343, 15.06, 15.777, 14.343, 15.06, 15.06, 13.626, 15.777, 15.777],
            "height": [25.998, 24.998, 24.998, 25.998, 25.817, 26.534, 26.534, 27.251, 26.534, 26.998],
        },
        "Hyperbolic": {
            "true_w": 15, "true_h": 30,
            "width":  [12.908, 12.191, 12.191, 11.474, 12.191, 12.908, 12.908, 14.343, 15.777, 14.343],
            "height": [22.998, 22.998, 22.998, 23.998, 22.948, 23.665, 23.665, 24.383, 25.817, 22.998],
        },
        "Tri. Prism": {
            "true_w": 15, "true_h": 30,
            "width":  [16.494, 17.211, 13.626, 12.908, 17.211, 19.363, 20.797, 17.928, 15.06, 13.626],
            "height": [28.998, 27.998, 27.998, 29.998, 27.968, 27.251, 27.968, 27.251, 25.817, 28.998],
        },
        "Sq. Pyramid": {
            "true_w": 15, "true_h": 30,
            "width":  [17.291, 18.646, 15.777, 15.777, 14.343, 15.06, 14.343, 15.06, 12.908, 12.908],
            "height": [27.998, 27.998, 26.998, 25.998, 25.817, 26.534, 25.1, 26.534, 25.1, 26.998],
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

DS_NAMES  = list(datasets.keys())       # ["MG1:1", "MG2:1"]
GEO_NAMES = list(list(datasets.values())[0].keys())
x = np.arange(len(GEO_NAMES))
BW = 0.35  # bar width

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Average % error  (1 row × 2 cols, one col per scale)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
for ax, ds in zip(axes, DS_NAMES):
    res   = all_results[ds]
    avg_w = [res[g]["width"]["avg_pct"]  for g in GEO_NAMES]
    avg_h = [res[g]["height"]["avg_pct"] for g in GEO_NAMES]
    ax.bar(x - BW/2, avg_w, BW, label="Width",  color=COLORS_W)
    ax.bar(x + BW/2, avg_h, BW, label="Height", color=COLORS_H)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylabel("Average % error")
    ax.set_title(f"{ds} — average % error vs. true")
    ax.legend()
fig.suptitle("Master Mold — Average percent error by geometry & scale", fontsize=13, fontweight="medium")
fig.tight_layout()
fig.savefig(OUT / "master_fig1_avg_pct_error.png")
plt.close(fig)
print("Saved master_fig1_avg_pct_error.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Bias  (1 × 2)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, ds in zip(axes, DS_NAMES):
    res    = all_results[ds]
    bias_w = [res[g]["width"]["bias"]  for g in GEO_NAMES]
    bias_h = [res[g]["height"]["bias"] for g in GEO_NAMES]
    ax.bar(x - BW/2, bias_w, BW,
           color=[COLORS_W if v >= 0 else RED for v in bias_w])
    ax.bar(x + BW/2, bias_h, BW,
           color=[COLORS_H if v >= 0 else RED for v in bias_h])
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Bias (mm)  [+ = over, − = under]")
    ax.set_title(f"{ds} — dimensional bias (mean − true)")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color=COLORS_W, label="Width over"),
    plt.Rectangle((0,0),1,1, color=COLORS_H, label="Height over"),
    plt.Rectangle((0,0),1,1, color=RED,      label="Under-sized"),
])
fig.suptitle("Master Mold — Dimensional bias by geometry & scale", fontsize=13, fontweight="medium")
fig.tight_layout()
fig.savefig(OUT / "master_fig2_bias.png")
plt.close(fig)
print("Saved master_fig2_bias.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Standard deviation  (1 × 2)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
for ax, ds in zip(axes, DS_NAMES):
    res  = all_results[ds]
    sd_w = [res[g]["width"]["sd"]  for g in GEO_NAMES]
    sd_h = [res[g]["height"]["sd"] for g in GEO_NAMES]
    ax.bar(x - BW/2, sd_w, BW, label="Width σ",  color=COLORS_W)
    ax.bar(x + BW/2, sd_h, BW, label="Height σ", color=COLORS_H)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Std dev σ (mm)")
    ax.set_title(f"{ds} — sample standard deviation")
    ax.legend()
fig.suptitle("Master Mold — Standard deviation by geometry & scale", fontsize=13, fontweight="medium")
fig.tight_layout()
fig.savefig(OUT / "master_fig3_std_dev.png")
plt.close(fig)
print("Saved master_fig3_std_dev.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-sample scatter  (2 rows × 5 cols: row=scale, col=geometry)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for row, ds in enumerate(DS_NAMES):
    res = all_results[ds]
    for col, gname in enumerate(GEO_NAMES):
        ax = axes[row][col]
        wv = res[gname]["width"]["values"]
        hv = res[gname]["height"]["values"]
        tw = res[gname]["width"]["true"]
        th = res[gname]["height"]["true"]
        xs = np.arange(1, len(wv) + 1)
        ax.scatter(xs, wv, color=COLORS_W, s=40, zorder=3, label="Width")
        ax.scatter(xs, hv, color=COLORS_H, s=40, zorder=3, label="Height")
        ax.axhline(tw, color=COLORS_W, linewidth=1.0, linestyle="--")
        ax.axhline(th, color=COLORS_H, linewidth=1.0, linestyle="--")
        ax.axhline(res[gname]["width"]["mean"],  color=COLORS_W, linewidth=0.8, linestyle=":")
        ax.axhline(res[gname]["height"]["mean"], color=COLORS_H, linewidth=0.8, linestyle=":")
        ax.set_title(f"{gname}\n({ds})", fontsize=8)
        ax.set_xlabel("Sample #", fontsize=8)
        ax.set_ylabel("mm", fontsize=8)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=7)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
fig.suptitle("Master Mold — Per-sample measurements  |  dashed = true, dotted = mean", fontsize=12, fontweight="medium")
fig.tight_layout()
fig.savefig(OUT / "master_fig4_per_sample_scatter.png")
plt.close(fig)
print("Saved master_fig4_per_sample_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-sample % error bars  (2 rows × 5 cols)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
for row, ds in enumerate(DS_NAMES):
    res = all_results[ds]
    for col, gname in enumerate(GEO_NAMES):
        ax = axes[row][col]
        pw = res[gname]["width"]["pct_errs"]
        ph = res[gname]["height"]["pct_errs"]
        xs = np.arange(1, len(pw) + 1)
        bw = 0.35
        ax.bar(xs - bw/2, pw, bw,
               color=[RED if v > 3 else AMBER if v > 1 else COLORS_W for v in pw])
        ax.bar(xs + bw/2, ph, bw,
               color=[RED if v > 3 else AMBER if v > 1 else COLORS_H for v in ph])
        ax.set_title(f"{gname}\n({ds})", fontsize=8)
        ax.set_xlabel("Sample", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=7)
        if col == 0:
            ax.set_ylabel("% error", fontsize=8)
fig.suptitle("Master Mold — Per-sample % error  |  Blue/green <1%  ·  Amber 1–3%  ·  Red >3%", fontsize=12, fontweight="medium")
fig.tight_layout()
fig.savefig(OUT / "master_fig5_pct_error_bars.png")
plt.close(fig)
print("Saved master_fig5_pct_error_bars.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Scale comparison: width bias MG1:1 vs MG2:1 side by side
# (uses normalised bias as % of true so the different true values are comparable)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, dim in zip(axes, ("width", "height")):
    bias_mg11 = [all_results["MG1:1"][g][dim]["bias"] / all_results["MG1:1"][g][dim]["true"] * 100
                 for g in GEO_NAMES]
    bias_mg21 = [all_results["MG2:1"][g][dim]["bias"] / all_results["MG2:1"][g][dim]["true"] * 100
                 for g in GEO_NAMES]
    ax.bar(x - BW/2, bias_mg11, BW, label="MG1:1", color="#85B7EB")
    ax.bar(x + BW/2, bias_mg21, BW, label="MG2:1", color=COLORS_W if dim == "width" else COLORS_H)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylabel("Relative bias (% of true)")
    ax.set_title(f"{dim.capitalize()} — relative bias: MG1:1 vs MG2:1")
    ax.legend()
fig.suptitle("Master Mold — Scale comparison: relative bias MG1:1 vs MG2:1", fontsize=13, fontweight="medium")
fig.tight_layout()
fig.savefig(OUT / "master_fig6_scale_comparison_bias.png")
plt.close(fig)
print("Saved master_fig6_scale_comparison_bias.png")

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