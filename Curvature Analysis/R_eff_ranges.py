import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

r_min_cutoff = 0.5
N = 4000
CELL = 15.0

ellipsoid_ratio = {
    "1:1": {"a": 15.0, "b": 30.0},
    "2:1": {"a": 15.0, "b": 60.0},
    "4:1": {"a": 15.0, "b": 120.0},
}
ellipsoid_scaled = {
    "2:1":           {"a": 15.0, "b":  30.0},
    "2:1 scaled 2x": {"a": 30.0, "b":  60.0},
    "2:1 scaled 4x": {"a": 60.0, "b": 120.0},
}

COLORS_RATIO  = {"1:1": "steelblue", "2:1": "darkorange", "4:1": "mediumpurple"}
COLORS_SCALED = {"2:1": "darkorange", "2:1 scaled 2x": "seagreen", "2:1 scaled 4x": "teal"}


def compute_ellipsoid(a, b):
    z = np.linspace(0.0, b, N)
    inside = np.clip(1.0 - (z / b)**2, 0.0, None)
    r = a * np.sqrt(inside)
    dr_dz   = np.gradient(r, z)
    d2r_dz2 = np.gradient(dr_dz, z)
    k_mer  = -d2r_dz2 / (1.0 + dr_dz**2)**1.5
    k_circ =  1.0 / (r * np.sqrt(1.0 + dr_dz**2))
    mask = r >= r_min_cutoff
    k_mer, k_circ = k_mer[mask], k_circ[mask]
    k1 = np.maximum(k_mer, k_circ)
    k2 = np.minimum(k_mer, k_circ)
    H  = np.abs((k1 + k2) / 2.0)
    return 1.0 / H  # R_eff profile


# Compute ranges
ratio_results = {}
for name, g in ellipsoid_ratio.items():
    R = compute_ellipsoid(g["a"], g["b"])
    ratio_results[name] = {"R_apex": R.min(), "R_base": R.max()}

scaled_results = {}
for name, g in ellipsoid_scaled.items():
    R = compute_ellipsoid(g["a"], g["b"])
    scaled_results[name] = {"R_apex": R.min(), "R_base": R.max()}


def plot_ranges(ax, results, colors, title):
    names = list(results.keys())
    y_pos = np.arange(len(names))

    for i, name in enumerate(names):
        r = results[name]
        c = colors[name]
        R_apex = r["R_apex"]
        R_base = r["R_base"]

        # Shaded range bar
        ax.barh(i, R_base - R_apex, left=R_apex, height=0.45,
                color=c, alpha=0.25, zorder=2)
        # Connecting line
        ax.plot([R_apex, R_base], [i, i], '-', color=c, lw=1.5, alpha=0.6, zorder=3)
        # Endpoint markers
        ax.plot(R_apex, i, 'o', color=c, ms=7, zorder=4)
        ax.plot(R_base, i, 's', color=c, ms=7, zorder=4, alpha=0.7)
        # Value labels
        ax.text(R_apex - 1.5, i, f"{R_apex:.1f}", ha='right', va='center',
                fontsize=7.5, color=c)
        ax.text(R_base + 1.5, i, f"{R_base:.1f}", ha='left', va='center',
                fontsize=7.5, color=c, alpha=0.85)

    # Cell diameter reference line
    ax.axvline(CELL, color='steelblue', lw=1.2, ls='--', alpha=0.7, zorder=5)
    ax.text(CELL + 0.5, len(names) - 0.35, f"cell ø\n{CELL} µm",
            fontsize=7.5, color='steelblue', va='top', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(r"$R_\mathrm{eff} = 1/H$  (µm)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', lw=0, ms=6, label='apex (sharpest)'),
        Line2D([0], [0], marker='s', color='gray', lw=0, ms=6, alpha=0.7, label='base (gentlest)'),
        Line2D([0], [0], color='steelblue', lw=1.2, ls='--', alpha=0.7,
               label=f'cell ø = {CELL} µm'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
fig.suptitle("Effective Radius of Curvature — Range across Surface",
             fontsize=11, fontweight='bold')

plot_ranges(axes[0], ratio_results, COLORS_RATIO,
            "Aspect Ratio Effect\n(fixed base radius a = 15 µm)")
plot_ranges(axes[1], scaled_results, COLORS_SCALED,
            "Scale Effect\n(2:1 shape, base radius 15 → 60 µm)")

plt.tight_layout()
plt.savefig("R_eff_ranges.png", dpi=150, bbox_inches='tight')
plt.show()