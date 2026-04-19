import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CHARACTERIZATION FIGURES
#
# Figure A: Ratio geometries  (1:1, 2:1, 4:1)  — shape effect
# Figure B: Scaled geometries (2:1, 2:1 2x, 2:1 4x) — size effect
#
# Each figure: 1x2 panel
#   Left:  Ellipsoid       K (solid) & H (dashed)
#   Right: Hyperbolic Cone K (solid) & H (dashed)
# ============================================================

r_min_cutoff = 0.5
N = 4000

# -----------------------------
# COLORS
# -----------------------------

COLORS_RATIO = {
    "1:1":  "steelblue",
    "2:1":  "darkorange",
    "4:1":  "mediumpurple",
}

COLORS_SCALED = {
    "2:1":           "darkorange",
    "2:1 scaled 2x": "seagreen",
    "2:1 scaled 4x": "teal",
}

# -----------------------------
# ELLIPSOID GEOMETRIES
# -----------------------------

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

# -----------------------------
# CONE GEOMETRIES
# -----------------------------

cone_ratio = {
    "1:1": {"P_0": (7.5,  0), "P_1": (2,  4), "P_2": (0,  15)},
    "2:1": {"P_0": (7.5,  0), "P_1": (2,  4), "P_2": (0,  30)},
    "4:1": {"P_0": (7.5,  0), "P_1": (2,  4), "P_2": (0,  60)},
}

cone_scaled = {
    "2:1":           {"P_0": (7.5,  0), "P_1": (2,   4), "P_2": (0,  30)},
    "2:1 scaled 2x": {"P_0": (15.0, 0), "P_1": (5,  10), "P_2": (0,  60)},
    "2:1 scaled 4x": {"P_0": (30.0, 0), "P_1": (10, 20), "P_2": (0, 120)},
}

# -----------------------------
# COMPUTE FUNCTIONS
# -----------------------------

def compute_ellipsoid(a, b, N=4000, r_min_cutoff=0.5):
    z = np.linspace(0.0, b, N)
    inside = np.clip(1.0 - (z / b)**2, 0.0, None)
    r = a * np.sqrt(inside)

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
    K_param = ((r_t * z_tt - z_t * r_tt) * z_t) / (r * (r_t**2 + z_t**2)**2)

    mask = r >= r_min_cutoff
    r, z, K_param = r[mask], z[mask], K_param[mask]

    sort_idx = np.argsort(z)
    z, r, K_param = z[sort_idx], r[sort_idx], K_param[sort_idx]

    # Compute derivatives in correct increasing-z order BEFORE flipping
    dr_dz   = np.gradient(r, z)
    d2r_dz2 = np.gradient(dr_dz, z)
    k_mer  = -d2r_dz2 / (1.0 + dr_dz**2)**1.5
    k_circ =  1.0 / (r * np.sqrt(1.0 + dr_dz**2))

    k1 = np.maximum(k_mer, k_circ)
    k2 = np.minimum(k_mer, k_circ)
    H  = (k1 + k2) / 2.0
    K  = -d2r_dz2 / (r * (1.0 + dr_dz**2)**2)

    # Flip: z_base=0 at base, z_base=max at tip, then normalize
    z_base = (z.max() - z)[::-1].copy()
    K  = K[::-1].copy()
    H  = H[::-1].copy()

    z_norm = z_base / z_base.max()
    return {"z_norm": z_norm, "K": K, "H": H}


# Pre-compute all results
ell_ratio_res  = {n: compute_ellipsoid(g["a"], g["b"]) for n, g in ellipsoid_ratio.items()}
ell_scaled_res = {n: compute_ellipsoid(g["a"], g["b"]) for n, g in ellipsoid_scaled.items()}
con_ratio_res  = {n: compute_cone(g["P_0"], g["P_1"], g["P_2"]) for n, g in cone_ratio.items()}
con_scaled_res = {n: compute_cone(g["P_0"], g["P_1"], g["P_2"]) for n, g in cone_scaled.items()}

# -----------------------------
# PLOT HELPER
# -----------------------------

def make_figure(ell_results, con_results, colors, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    ax_ell = axes[0]
    ax_con = axes[1]

    for name, d in ell_results.items():
        c = colors[name]
        ax_ell.plot(d["z_norm"], d["K"], color=c, linewidth=2,
                    label=f"{name} — K")
        ax_ell.plot(d["z_norm"], d["H"], color=c, linewidth=2,
                    linestyle='--', alpha=0.7, label=f"{name} — H")

    for name, d in con_results.items():
        c = colors[name]
        ax_con.plot(d["z_norm"], d["K"], color=c, linewidth=2,
                    label=f"{name} — K")
        ax_con.plot(d["z_norm"], d["H"], color=c, linewidth=2,
                    linestyle='--', alpha=0.7, label=f"{name} — H")

    # Shared formatting
    for ax, surface, xlabel in [
        (ax_ell, "Ellipsoid",       "normalized height (0 = base, 1 = apex)"),
        (ax_con, "Hyperbolic Cone", "normalized height (0 = base, 1 = tip)"),
    ]:
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("curvature (1/µm  or  1/µm²)", fontsize=8)
        ax.set_title(f"{surface} — Gaussian K (solid) & Mean H (dashed)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, linestyle='--', alpha=0.3)

    # Annotate K=0 line on cone panel (after data so ylim is set)
    ymin_con = ax_con.get_ylim()[0]
    ax_con.annotate("K = 0  (cylindrical)",
                    xy=(0.6, 0),
                    xytext=(0.35, ymin_con * 0.4),
                    fontsize=7, color='gray',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


# ==============================
# FIGURE A: Ratio geometries
# ==============================

make_figure(
    ell_ratio_res, con_ratio_res,
    COLORS_RATIO,
    "Curvature Characterization — Aspect Ratio Effect",
    "characterization_A_ratio.png"
)

# ==============================
# FIGURE B: Scaled geometries
# ==============================

make_figure(
    ell_scaled_res, con_scaled_res,
    COLORS_SCALED,
    "Curvature Characterization — Scale Effect",
    "characterization_B_scaled.png"
)