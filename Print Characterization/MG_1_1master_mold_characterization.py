"""
Print Accuracy Analysis — Multi-Geometry
Computes percent error, bias, mean, and std dev vs. true dimension (15 mm)
and saves figures to the output directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
TRUE = 15.0

geometries = {
    "Hemi-ellipsoid": {
        "width":  [15.06, 14.343, 15.06, 14.343, 14.343, 15.06, 14.353, 14.343, 15.06, 13.626],
        "height": [15.999, 16.999, 16.238, 13.999, 19.363, 17.928, 17.549, 17.928, 18.646, 17.23],
    },
    "Cone": {
        "width":  [12.908, 12.191, 13.626, 13.626, 12.908, 13.626, 11.474, 13.626, 14.343, 11.514],
        "height": [12.999, 12.999, 13.999, 12.951, 13.626, 12.908, 13.626, 13.077, 13.626, 13.999],
    },
    "Hyperbolic": {
        "width":  [12.191, 12.191, 12.191, 12.908, 12.908, 13.626, 11.474, 12.908, 13.626, 10.757],
        "height": [10.999, 11.231, 10.999, 10.999, 10.575, 10.757, 11.474, 11.474, 11.474, 10.999],
    },
    "Triangular Prism": {
        "width":  [12.191, 12.908, 12.908, 12.908, 12.908, 13.626, 12.908, 13.626, 12.908, 12.191],
        "height": [15.999, 15.999, 14.999, 14.999, 15.06,  15.06,  15.777, 15.777, 15.06,  14.999],
    },
    "Square Pyramid": {
        "width":  [14.343, 13.626, 13.626, 14.343, 13.626, 14.343, 12.908, 14.343, 14.343, 12.908],
        "height": [14.999, 15.999, 15.999, 15.999, 15.777, 15.777, 16.494, 16.494, 15.06,  16.999],
    },
}

# ── Stats helper ──────────────────────────────────────────────────────────────
def compute_stats(values):
    arr = np.array(values)
    mean   = arr.mean()
    sd     = arr.std(ddof=1)          # sample std dev (matches Excel STDEV)
    bias   = mean - TRUE
    pct_errs = np.abs(arr - TRUE) / TRUE * 100
    avg_pct  = pct_errs.mean()
    return dict(mean=mean, sd=sd, bias=bias, avg_pct=avg_pct,
                pct_errs=pct_errs, values=arr)

results = {
    name: {dim: compute_stats(data[dim]) for dim in ("width", "height")}
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
W = 0.35   # bar width

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Average % error (width & height) per geometry
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
ax.set_title("MG1:1 Average percent error vs. true (15 mm)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "MG1_1fig1_avg_pct_error.png")
plt.close(fig)
print("Saved MG1_1fig1_avg_pct_error.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Bias (mean − 15 mm)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bias_w = [results[g]["width"]["bias"]  for g in GEO_NAMES]
bias_h = [results[g]["height"]["bias"] for g in GEO_NAMES]

bar_w = ax.bar(x - W/2, bias_w, W, label="Width",
               color=[COLORS_W if v >= 0 else RED for v in bias_w])
bar_h = ax.bar(x + W/2, bias_h, W, label="Height",
               color=[COLORS_H if v >= 0 else RED for v in bias_h])
ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(GEO_NAMES, rotation=15, ha="right")
ax.set_ylabel("Bias (mm)  [+ = over, − = under]")
ax.set_title("MG1:1 Dimensional bias per geometry")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color=COLORS_W, label="Width (+ over)"),
    plt.Rectangle((0,0),1,1, color=COLORS_H, label="Height (+ over)"),
    plt.Rectangle((0,0),1,1, color=RED,      label="Under-sized"),
])
fig.tight_layout()
fig.savefig(OUT / "MG1_1fig2_bias.png")
plt.close(fig)
print("Saved MG1_1fig2_bias.png")

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
ax.set_title("MG1:1 Sample standard deviation per geometry")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "MG1_1fig3_std_dev.png")
plt.close(fig)
print("Saved MG1_1fig3_std_dev.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-sample scatter plots (one subplot per geometry)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, gname in enumerate(GEO_NAMES):
    ax = axes[i]
    wvals = results[gname]["width"]["values"]
    hvals = results[gname]["height"]["values"]
    n = len(wvals)
    xs = np.arange(1, n + 1)

    ax.scatter(xs, wvals, color=COLORS_W, zorder=3, label="Width",  s=40)
    ax.scatter(xs, hvals, color=COLORS_H, zorder=3, label="Height", s=40)
    ax.axhline(TRUE, color=RED,   linewidth=1.2, linestyle="--", label="True (15 mm)")
    ax.axhline(results[gname]["width"]["mean"],  color=COLORS_W, linewidth=0.8, linestyle=":")
    ax.axhline(results[gname]["height"]["mean"], color=COLORS_H, linewidth=0.8, linestyle=":")
    ax.set_title(gname, fontsize=11)
    ax.set_xlabel("Sample #")
    ax.set_ylabel("mm")
    ax.set_xticks(xs)
    if i == 0:
        ax.legend(fontsize=8)

axes[-1].set_visible(False)
fig.suptitle("MG1:1 Per-sample measurements vs. true (15 mm) — dashed red = true, dotted = mean", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "MG1_1fig4_per_sample_scatter.png")
plt.close(fig)
print("Saved MG1_1fig4_per_sample_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-sample % error heatmap style (grouped bars per geometry)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)

for i, gname in enumerate(GEO_NAMES):
    ax = axes[i]
    pct_w = results[gname]["width"]["pct_errs"]
    pct_h = results[gname]["height"]["pct_errs"]
    n = len(pct_w)
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

fig.suptitle("MG1:1 Per-sample % error  |  Blue/green < 1%  ·  Amber 1–3%  ·  Red > 3%", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "MG1_1fig5_pct_error_bars.png")
plt.close(fig)
print("Saved MG1_1fig5_pct_error_bars.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table to console
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"{'Geometry':<18} {'Dim':<8} {'Mean':>8} {'Std dev':>8} {'Bias':>8} {'Avg %err':>9}")
print("="*72)
for gname in GEO_NAMES:
    for dim in ("width", "height"):
        s = results[gname][dim]
        print(f"{gname if dim=='width' else '':<18} {dim:<8} "
              f"{s['mean']:>8.3f} {s['sd']:>8.4f} "
              f"{s['bias']:>+8.4f} {s['avg_pct']:>8.2f}%")
    print("-"*72)

print(f"\nAll figures saved to: {OUT.resolve()}/")