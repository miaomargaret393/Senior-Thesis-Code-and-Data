"""
Print Accuracy Analysis — Multi-Geometry MG2:1
Computes percent error, bias, mean, and std dev vs. true dimensions.
Saves figures to the same directory as this script.

True dimensions vary by geometry:
  HemiEllipsoid : width=30, height=30
  Cone          : width=15, height=30
  Hyperbolic    : width=15, height=30
  Tri. Prism    : width=15, height=30
  Sq. Pyramid   : width=15, height=30
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
geometries = {
    "Hemi-ellipsoid": {
        "true_w": 30, "true_h": 30,
        "width":  [31.554, 31.554, 30.837, 29.403, 29.403, 28.523, 29.061, 28.523, 29.403, 30.675],
        "height": [27.998, 28.998, 27.998, 27.878, 29.998, 27.838, 26.748, 30.998, 28.685, 27.998],
    },
    "Cone": {
        "true_w": 15, "true_h": 30,
        "width":  [15.777, 14.343, 15.06, 15.777, 14.343, 15.06, 15.06, 13.626, 15.777, 15.777],
        "height": [25.998, 24.998, 24.998, 25.998, 25.817, 26.534, 26.534, 17.251, 26.534, 26.998],
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
}

# ── Stats helper ──────────────────────────────────────────────────────────────
def compute_stats(values, true_val):
    arr = np.array(values)
    mean     = arr.mean()
    sd       = arr.std(ddof=1)
    bias     = mean - true_val
    pct_errs = np.abs(arr - true_val) / true_val * 100
    avg_pct  = pct_errs.mean()
    return dict(mean=mean, sd=sd, bias=bias, avg_pct=avg_pct,
                pct_errs=pct_errs, values=arr, true=true_val)

results = {
    name: {
        "width":  compute_stats(data["width"],  data["true_w"]),
        "height": compute_stats(data["height"], data["true_h"]),
    }
    for name, data in geometries.items()
}

# ── Style ─────────────────────────────────────────────────────────────────────
GEO_NAMES  = list(geometries.keys())
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

x = np.arange(len(GEO_NAMES))
W = 0.35

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Average % error per geometry
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
avg_pct_w = [results[g]["width"]["avg_pct"]  for g in GEO_NAMES]
avg_pct_h = [results[g]["height"]["avg_pct"] for g in GEO_NAMES]

ax.bar(x - W/2, avg_pct_w, W, label="Width",  color=COLORS_W)
ax.bar(x + W/2, avg_pct_h, W, label="Height", color=COLORS_H)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.set_ylabel("Average % error")
ax.set_title("MG2:1 Average percent error vs. true dimensions (per geometry)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "MG2_1_fig1_avg_pct_error.png")
plt.close(fig)
print("Saved MG2_1_fig1_avg_pct_error.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Bias (mean − true)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bias_w = [results[g]["width"]["bias"]  for g in GEO_NAMES]
bias_h = [results[g]["height"]["bias"] for g in GEO_NAMES]

ax.bar(x - W/2, bias_w, W,
       color=[COLORS_W if v >= 0 else RED for v in bias_w])
ax.bar(x + W/2, bias_h, W,
       color=[COLORS_H if v >= 0 else RED for v in bias_h])
ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
ax.set_ylabel("Bias (mm)  [+ = over, − = under]")
ax.set_title("MG2:1 Dimensional bias per geometry (mean − true)")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color=COLORS_W, label="Width (+ over)"),
    plt.Rectangle((0,0),1,1, color=COLORS_H, label="Height (+ over)"),
    plt.Rectangle((0,0),1,1, color=RED,      label="Under-sized"),
])
fig.tight_layout()
fig.savefig(OUT / "MG2_1_fig2_bias.png")
plt.close(fig)
print("Saved MG2_1_fig2_bias.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Standard deviation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sd_w = [results[g]["width"]["sd"]  for g in GEO_NAMES]
sd_h = [results[g]["height"]["sd"] for g in GEO_NAMES]

ax.bar(x - W/2, sd_w, W, label="Width σ",  color=COLORS_W)
ax.bar(x + W/2, sd_h, W, label="Height σ", color=COLORS_H)
ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
ax.set_ylabel("Std dev σ (mm)")
ax.set_title("MG2:1 Sample standard deviation per geometry")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "MG2_1_fig3_std_dev.png")
plt.close(fig)
print("Saved MG2_1_fig3_std_dev.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-sample scatter plots
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, gname in enumerate(GEO_NAMES):
    ax = axes[i]
    wvals  = results[gname]["width"]["values"]
    hvals  = results[gname]["height"]["values"]
    true_w = results[gname]["width"]["true"]
    true_h = results[gname]["height"]["true"]
    n  = len(wvals)
    xs = np.arange(1, n + 1)

    ax.scatter(xs, wvals, color=COLORS_W, zorder=3, label="Width",  s=40)
    ax.scatter(xs, hvals, color=COLORS_H, zorder=3, label="Height", s=40)
    ax.axhline(true_w, color=COLORS_W, linewidth=1.0, linestyle="--")
    ax.axhline(true_h, color=COLORS_H, linewidth=1.0, linestyle="--",
               label=f"True W={true_w} / H={true_h}")
    ax.axhline(results[gname]["width"]["mean"],  color=COLORS_W, linewidth=0.8, linestyle=":")
    ax.axhline(results[gname]["height"]["mean"], color=COLORS_H, linewidth=0.8, linestyle=":")
    ax.set_title(gname, fontsize=11)
    ax.set_xlabel("Sample #")
    ax.set_ylabel("mm")
    ax.set_xticks(xs)
    if i == 0:
        ax.legend(fontsize=7)

axes[-1].set_visible(False)
fig.suptitle("MG2:1 Per-sample measurements vs. true — dashed = true, dotted = mean", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "MG2_1_fig4_per_sample_scatter.png")
plt.close(fig)
print("Saved MG2_1_fig4_per_sample_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-sample % error bars
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)

for i, gname in enumerate(GEO_NAMES):
    ax = axes[i]
    pct_w = results[gname]["width"]["pct_errs"]
    pct_h = results[gname]["height"]["pct_errs"]
    n  = len(pct_w)
    xs = np.arange(1, n + 1)
    bw = 0.35

    ax.bar(xs - bw/2, pct_w, bw,
           color=[RED if v > 3 else AMBER if v > 1 else COLORS_W for v in pct_w])
    ax.bar(xs + bw/2, pct_h, bw,
           color=[RED if v > 3 else AMBER if v > 1 else COLORS_H for v in pct_h])
    ax.set_title(gname, fontsize=9)
    ax.set_xlabel("Sample")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, fontsize=7)
    if i == 0:
        ax.set_ylabel("% error")

fig.suptitle("MG2:1 Per-sample % error  |  Blue/green < 1%  ·  Amber 1–3%  ·  Red > 3%", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "MG2_1_fig5_pct_error_bars.png")
plt.close(fig)
print("Saved MG2_1_fig5_pct_error_bars.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'Geometry':<18} {'Dim':<8} {'True':>6} {'Mean':>8} {'Std dev':>8} {'Bias':>8} {'Avg %err':>9}")
print("="*80)
for gname in GEO_NAMES:
    for dim in ("width", "height"):
        s = results[gname][dim]
        print(f"{gname if dim=='width' else '':<18} {dim:<8} "
              f"{s['true']:>6.0f} {s['mean']:>8.3f} {s['sd']:>8.4f} "
              f"{s['bias']:>+8.4f} {s['avg_pct']:>8.2f}%")
    print("-"*80)

print(f"\nAll figures saved to: {OUT.resolve()}/")