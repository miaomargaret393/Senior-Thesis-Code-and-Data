import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ============================================================
# CURVATURE + NUC/HEIGHT OVERLAY
#
# Replicates Figure A (ratio) and Figure B (scaled) from the
# characterisation script, adding shaded Nuc/Height % bands
# on the hemi-ellipsoid and hyperbolic-cone panels.
# ============================================================

r_min_cutoff = 0.5
N = 4000

# -----------------------------------------------------------
# COLORS
# -----------------------------------------------------------
COLORS_RATIO = {
    "1:1": "steelblue",
    "2:1": "darkorange",
}
COLORS_SCALED = {
    "2:1":            "darkorange",
    "2:1 scaled 2x":  "seagreen",
    "2:1 scaled 4x":  "teal",
}

# -----------------------------------------------------------
# NUC/HEIGHT % DATA  (lo, hi) per geometry × AR
#   Key format: (geometry_label, AR_label)
# -----------------------------------------------------------
# For ratio figure we have 1:1 and 2:1 (4:1 not measured → skip band)
# For scaled figure the base geometry is 2:1; scaled variants
# share the same biological data (same AR, same cell biology),
# so we draw one band for 2:1.

NUC_HEIGHT = {
    # Hemi-ellipsoid
    ("Hemi-Ellipsoid", "2:1"): (0.89, 0.93),
    ("Hemi-Ellipsoid", "1:1"): (0.70, 0.75),
    # Hyperbolic Cone
    ("Hyperbolic Cone", "2:1"): (0.36, 0.39),
    ("Hyperbolic Cone", "1:1"): (0.39, 0.62),
}

# -----------------------------------------------------------
# ELLIPSOID GEOMETRIES
# -----------------------------------------------------------
ellipsoid_ratio = {
    "1:1": {"a": 15.0, "b": 30.0},
    "2:1": {"a": 15.0, "b": 60.0},
}
ellipsoid_scaled = {
    "2:1":            {"a": 15.0, "b":  30.0},
    "2:1 scaled 2x":  {"a": 30.0, "b":  60.0},
    "2:1 scaled 4x":  {"a": 60.0, "b": 120.0},
}

# -----------------------------------------------------------
# CONE GEOMETRIES
# -----------------------------------------------------------
cone_ratio = {
    "1:1": {"P_0": (7.5,  0), "P_1": (2,  4), "P_2": (0,  15)},
    "2:1": {"P_0": (7.5,  0), "P_1": (2,  4), "P_2": (0,  30)},
}
cone_scaled = {
    "2:1":            {"P_0": (7.5,  0), "P_1": (2,   4), "P_2": (0,  30)},
    "2:1 scaled 2x":  {"P_0": (15.0, 0), "P_1": (5,  10), "P_2": (0,  60)},
    "2:1 scaled 4x":  {"P_0": (30.0, 0), "P_1": (10, 20), "P_2": (0, 120)},
}

# -----------------------------------------------------------
# COMPUTE FUNCTIONS  (unchanged from original)
# -----------------------------------------------------------
def compute_ellipsoid(a, b, N=4000, r_min_cutoff=0.5):
    z = np.linspace(0.0, b, N)
    r = a * np.sqrt(np.clip(1.0 - (z / b)**2, 0.0, None))
    dr_dz   = np.gradient(r, z)
    d2r_dz2 = np.gradient(dr_dz, z)
    K      = -d2r_dz2 / (r * (1.0 + dr_dz**2)**2)
    k_mer  = -d2r_dz2 / (1.0 + dr_dz**2)**1.5
    k_circ =  1.0 / (r * np.sqrt(1.0 + dr_dz**2))
    mask = r >= r_min_cutoff
    z, K = z[mask], K[mask]
    k_mer, k_circ = k_mer[mask], k_circ[mask]
    k1 = np.maximum(k_mer, k_circ)
    k2 = np.minimum(k_mer, k_circ)
    H  = (k1 + k2) / 2.0
    z_norm = z / z.max()
    return {"z_norm": z_norm, "K": K, "H": H}


def compute_cone(P_0, P_1, P_2, w=0.75, N=4000, r_min_cutoff=0.5):
    t = np.linspace(1e-4, 0.999, N)
    D = (1-t)**2 + 2*w*(1-t)*t + t**2
    r = ((1-t)**2*P_0[0] + 2*w*(1-t)*t*P_1[0] + t**2*P_2[0]) / D
    z = ((1-t)**2*P_0[1] + 2*w*(1-t)*t*P_1[1] + t**2*P_2[1]) / D
    def d_dt(arr): return np.gradient(arr, t)
    r_t = d_dt(r); z_t = d_dt(z)
    r_tt = d_dt(r_t); z_tt = d_dt(z_t)
    mask = r >= r_min_cutoff
    r, z = r[mask], z[mask]
    r_t, z_t = r_t[mask], z_t[mask]
    sort_idx = np.argsort(z)
    z, r = z[sort_idx], r[sort_idx]
    dr_dz   = np.gradient(r, z)
    d2r_dz2 = np.gradient(dr_dz, z)
    k_mer  = -d2r_dz2 / (1.0 + dr_dz**2)**1.5
    k_circ =  1.0 / (r * np.sqrt(1.0 + dr_dz**2))
    k1 = np.maximum(k_mer, k_circ)
    k2 = np.minimum(k_mer, k_circ)
    H  = (k1 + k2) / 2.0
    K  = -d2r_dz2 / (r * (1.0 + dr_dz**2)**2)
    z_base = (z.max() - z)[::-1].copy()
    K  = K[::-1].copy()
    H  = H[::-1].copy()
    z_norm = z_base / z_base.max()
    return {"z_norm": z_norm, "K": K, "H": H}


# Pre-compute
ell_ratio_res  = {n: compute_ellipsoid(g["a"], g["b"]) for n, g in ellipsoid_ratio.items()}
ell_scaled_res = {n: compute_ellipsoid(g["a"], g["b"]) for n, g in ellipsoid_scaled.items()}
con_ratio_res  = {n: compute_cone(g["P_0"], g["P_1"], g["P_2"]) for n, g in cone_ratio.items()}
con_scaled_res = {n: compute_cone(g["P_0"], g["P_1"], g["P_2"]) for n, g in cone_scaled.items()}

# -----------------------------------------------------------
# BAND COLORS  (one per AR, matched to curve color)
# -----------------------------------------------------------
BAND_ALPHA = 0.13   # fill opacity for the shaded band
BAND_LINE_ALPHA = 0.55  # opacity for dashed boundary lines

# For ratio figure: AR colors match COLORS_RATIO
# For scaled figure: all variants share 2:1 biology, use orange


def add_nuc_bands(ax, geom_label, ar_color_map, results_dict, band_alpha=BAND_ALPHA):
    """
    Shade Nuc/Height % range on ax for each AR that has data.
    geom_label: "Hemi-Ellipsoid" or "Hyperbolic Cone"
    ar_color_map: dict  AR_label -> color
    results_dict: dict  AR_label -> {"z_norm":…}  (only used for ylim)
    """
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin

    for ar_label, color in ar_color_map.items():
        key = (geom_label, ar_label)
        if key not in NUC_HEIGHT:
            continue
        lo, hi = NUC_HEIGHT[key]

        # Vertical span (shaded column across full y range)
        ax.axvspan(lo, hi,
                   color=color, alpha=band_alpha, linewidth=0)

        # Dashed boundary lines
        for x in (lo, hi):
            ax.axvline(x, color=color, alpha=BAND_LINE_ALPHA,
                       linewidth=1.0, linestyle='--', zorder=2)

        # Small annotation at the top of the band
        mid = (lo + hi) / 2.0
        ax.text(mid, ymax - 0.03 * y_span,
                f"{ar_label}\n{int(lo*100)}–{int(hi*100)}%",
                ha='center', va='top', fontsize=6,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))


# -----------------------------------------------------------
# MAIN FIGURE BUILDER
# -----------------------------------------------------------
def make_figure(ell_results, con_results, colors, title, filename,
                ell_geom_label="Hemi-Ellipsoid",
                con_geom_label="Hyperbolic Cone"):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    ax_ell = axes[0]
    ax_con = axes[1]

    # -- plot curvature curves first --
    for name, d in ell_results.items():
        c = colors[name]
        ax_ell.plot(d["z_norm"], d["K"], color=c, linewidth=2,   label=f"{name} — K")
        ax_ell.plot(d["z_norm"], d["H"], color=c, linewidth=2,
                    linestyle='--', alpha=0.7, label=f"{name} — H")

    for name, d in con_results.items():
        c = colors[name]
        ax_con.plot(d["z_norm"], d["K"], color=c, linewidth=2,   label=f"{name} — K")
        ax_con.plot(d["z_norm"], d["H"], color=c, linewidth=2,
                    linestyle='--', alpha=0.7, label=f"{name} — H")

    # -- shared formatting (must set before bands so ylim is fixed) --
    for ax, surface, xlabel in [
        (ax_ell, ell_geom_label,  "normalised height  (0 = base, 1 = apex)"),
        (ax_con, con_geom_label,  "normalised height  (0 = base, 1 = tip)"),
    ]:
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("curvature  (1/µm  or  1/µm²)", fontsize=8)
        ax.set_title(f"{surface} — K (solid) & H (dashed)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, 1)

    # -- freeze ylim before adding bands --
    ax_ell.autoscale(enable=True, axis='y')
    ax_con.autoscale(enable=True, axis='y')
    fig.canvas.draw()   # forces ylim to settle

    # -- add nuc/height bands --
    #    Build a mapping AR_label -> color from the colors dict
    #    Only keep ARs that are pure "1:1" or "2:1" (not scaled variants)
    ar_color_map = {k: v for k, v in colors.items()
                    if k in ("1:1", "2:1", "4:1")}

    add_nuc_bands(ax_ell, ell_geom_label, ar_color_map, ell_results)
    add_nuc_bands(ax_con, con_geom_label, ar_color_map, con_results)

    # -- K=0 annotation on cone panel --
    ymin_con = ax_con.get_ylim()[0]
    ax_con.annotate("K = 0  (cylindrical)",
                    xy=(0.6, 0),
                    xytext=(0.35, ymin_con * 0.4),
                    fontsize=7, color='gray',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # -- legend: curves + band legend --
    for ax in (ax_ell, ax_con):
        handles, labels = ax.get_legend_handles_labels()
        # Append band legend entries for ARs that have data
        for ar_label, color in ar_color_map.items():
            if any((g, ar_label) in NUC_HEIGHT for g in (ell_geom_label, con_geom_label)):
                patch = mpatches.Patch(
                    facecolor=color, alpha=0.35,
                    edgecolor=color, linewidth=0.8,
                    label=f"Nuc/Height {ar_label}")
                handles.append(patch)
                labels.append(f"Nuc/Height {ar_label}")
        ax.legend(handles=handles, labels=labels,
                  fontsize=6.5, loc="upper left",
                  framealpha=0.85, edgecolor='none')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


# ============================================================
# FIGURE A — Ratio geometries  (1:1, 2:1, 4:1)
# ============================================================
make_figure(
    ell_ratio_res, con_ratio_res,
    COLORS_RATIO,
    "Curvature + Nuc/Height % — Aspect Ratio Effect",
    "overlay_A_ratio.png",
)

# ============================================================
# FIGURE B — Scaled geometries
# For the scaled figure only 2:1 has biological data,
# so only one band appears on each panel.
# ============================================================
make_figure(
    ell_scaled_res, con_scaled_res,
    COLORS_SCALED,
    "Curvature + Nuc/Height % — Scale Effect",
    "overlay_B_scaled.png",
)