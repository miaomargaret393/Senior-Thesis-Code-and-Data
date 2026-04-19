import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# GEOMETRY DEFINITIONS
# -----------------------------

geometries = {
    "1:1": {
        "P_0": (7.5, 0),
        "P_2": (0, 15),
        "P_1": (2, 4),
    },
    "2:1": {
        "P_0": (7.5, 0),
        "P_2": (0, 30),
        "P_1": (2, 4),
    },
    "2:1 scaled 2x": {
        "P_0": (15, 0),
        "P_2": (0, 60),
        "P_1": (5, 10),
    },
    "2:1 scaled 4x": {
        "P_0": (30, 0),
        "P_2": (0, 120),
        "P_1": (10, 20),
    },
    "4:1": {
        "P_0": (7.5, 0),
        "P_2": (0, 60),
        "P_1": (2, 4),
    },
    "6:1": {
        "P_0": (7.5, 0),
        "P_2": (0, 90),
        "P_1": (2, 4),
    },
}

# Groups for per-geometry figures
RATIO_GEOS  = ["1:1", "2:1", "4:1"]
SCALED_GEOS = ["2:1 scaled 2x", "2:1 scaled 4x"]

# -----------------------------
# SHARED PARAMETERS
# -----------------------------

w = 0.75
r_min_cutoff = 0.5
N = 2000

COLORS = {
    "1:1":           {"K": "steelblue",    "H": "steelblue",    "k1": "steelblue",    "k2": "steelblue"},
    "2:1":           {"K": "darkorange",   "H": "darkorange",   "k1": "darkorange",   "k2": "darkorange"},
    "2:1 scaled 2x": {"K": "seagreen",     "H": "seagreen",     "k1": "seagreen",     "k2": "seagreen"},
    "2:1 scaled 4x": {"K": "teal",         "H": "teal",         "k1": "teal",         "k2": "teal"},
    "4:1":           {"K": "mediumpurple", "H": "mediumpurple", "k1": "mediumpurple", "k2": "mediumpurple"},
    "6:1":           {"K": "crimson",      "H": "crimson",      "k1": "crimson",      "k2": "crimson"},
}

# -----------------------------
# COMPUTE CURVATURES
# -----------------------------

def compute_curvature(P_0, P_1, P_2, w=0.75, N=2000, r_min_cutoff=0.5):
    t = np.linspace(1e-4, 0.999, N)
    D = (1 - t)**2 + 2*w*(1 - t)*t + t**2
    r = ((1 - t)**2 * P_0[0] + 2*w*(1 - t)*t * P_1[0] + t**2 * P_2[0]) / D
    z = ((1 - t)**2 * P_0[1] + 2*w*(1 - t)*t * P_1[1] + t**2 * P_2[1]) / D

    def d_dt(arr): return np.gradient(arr, t)

    r_t = d_dt(r); z_t = d_dt(z)
    r_tt = d_dt(r_t); z_tt = d_dt(z_t)

    K = ((r_t * z_tt - z_t * r_tt) * z_t) / (r * (r_t**2 + z_t**2)**2)

    mask = r >= r_min_cutoff
    r, z, K = r[mask], z[mask], K[mask]

    # Sort by increasing z (tip at low z, base at high z for this parametric curve)
    sort_idx = np.argsort(z)
    z, r, K = z[sort_idx], r[sort_idx], K[sort_idx]

    # Compute derivatives in correct increasing-z order BEFORE flipping axis
    dr_dz   = np.gradient(r, z)
    d2r_dz2 = np.gradient(dr_dz, z)
    k_mer  = -d2r_dz2 / (1.0 + dr_dz**2)**1.5
    k_circ =  1.0 / (r * np.sqrt(1.0 + dr_dz**2))

    k1 = np.maximum(k_mer, k_circ)
    k2 = np.minimum(k_mer, k_circ)
    H  = (k1 + k2) / 2.0
    dK_dz = np.gradient(K, z)

    # Flip axis for display: z_base=0 at cone base, z_base=max at tip
    z_base = z.max() - z
    z_base  = z_base[::-1].copy()
    r       = r[::-1].copy()
    K       = K[::-1].copy()
    k1      = k1[::-1].copy()
    k2      = k2[::-1].copy()
    H       = H[::-1].copy()
    dK_dz   = dK_dz[::-1].copy()

    return {"z_base": z_base, "K": K, "H": H, "k1": k1, "k2": k2, "dK_dz": dK_dz}

results = {}
for name, g in geometries.items():
    results[name] = compute_curvature(g["P_0"], g["P_1"], g["P_2"])

# -----------------------------
# SUMMARY TABLE
# -----------------------------

print("\n==== CURVATURE SUMMARY TABLE ====\n")
print(f"{'Geometry':<18} {'K_min':>12} {'K_max':>12} {'H_max':>12} {'H_min':>12} {'k1_max':>12} {'k2_min':>12}")
print("-" * 90)
for name, d in results.items():
    print(f"{name:<18} {d['K'].min():>12.4e} {d['K'].max():>12.4e} "
          f"{d['H'].max():>12.4e} {d['H'].min():>12.4e} "
          f"{d['k1'].max():>12.4e} {d['k2'].min():>12.4e}")

# -----------------------------
# FUSION / FIJI THRESHOLDS (per geometry)
# -----------------------------

def fusion_thresholds(name, d):
    z      = d["z_base"]
    K      = d["K"]
    k1     = d["k1"]
    k2     = d["k2"]

    eps      = 1e-12
    abs_k1   = np.abs(k1)
    abs_k2   = np.abs(k2)

    R1 = 1.0 / (abs_k1 + eps)
    R2 = 1.0 / (abs_k2 + eps)

    def to_mm_principal(v): return v * 1000.0
    def to_mm2_gaussian(v): return v * 1e6

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    print(f"\n  Gaussian K extrema:")
    print(f"    max K : {K.max():.4e} 1/µm²  →  {to_mm2_gaussian(K.max()):.4e} 1/mm²")
    print(f"    min K : {K.min():.4e} 1/µm²  →  {to_mm2_gaussian(K.min()):.4e} 1/mm²")

    print(f"\n  Principal k1 (meridional max):")
    print(f"    max k1: {k1.max():.4e} 1/µm   →  {to_mm_principal(k1.max()):.4e} 1/mm")
    print(f"    min k1: {k1.min():.4e} 1/µm   →  {to_mm_principal(k1.min()):.4e} 1/mm")

    print(f"\n  Principal k2 (circumferential min):")
    print(f"    max k2: {k2.max():.4e} 1/µm   →  {to_mm_principal(k2.max()):.4e} 1/mm")
    print(f"    min k2: {k2.min():.4e} 1/µm   →  {to_mm_principal(k2.min()):.4e} 1/mm")

    R1_min = np.min(R1)
    R1_rob = np.quantile(R1, 0.05)
    print(f"\n  Radius-of-curvature (k1 direction):")
    print(f"    tightest R1         : {R1_min:.3f} µm")
    print(f"    robust tight R1 (5%): {R1_rob:.3f} µm")

    ratio         = abs_k2 / (abs_k1 + eps)
    ratio_cutoff  = 0.25
    k2_mag_cutoff = np.quantile(abs_k2, 0.90)
    both_mask     = (ratio >= ratio_cutoff) & (abs_k2 >= k2_mag_cutoff)

    if np.any(both_mask):
        z_zone      = z[both_mask]
        R2_zone     = R2[both_mask]
        R2_thresh   = np.median(R2_zone)
        R2_min_zone = np.min(R2_zone)
        print(f"\n  'Both-directions bending' zone (k2 zone):")
        print(f"    z-range : {z_zone.min():.2f} → {z_zone.max():.2f} µm  ← use in Fiji")
        print(f"    median R2 (Fusion threshold): {R2_thresh:.3f} µm")
        print(f"    tightest R2 in zone         : {R2_min_zone:.3f} µm")
    else:
        print(f"\n  No clear both-directions bending zone found.")
        print(f"  Try lowering ratio_cutoff (currently {ratio_cutoff}).")

print("\n\n==== FUSION / FIJI THRESHOLD REPORT ====")
for name, d in results.items():
    fusion_thresholds(name, d)

# ==============================
# FIGURE 1: OVERLAID COMPARISONS (all geometries)
# Sized for 8.5 x 11 page (portrait)
# ==============================

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
fig.suptitle("Hyperbolic Cone Curvature Comparison", fontsize=11, fontweight='bold')

ax_KH  = axes[0]
ax_kpc = axes[1]

for name, d in results.items():
    c = COLORS[name]["K"]
    z_norm = d["z_base"] / d["z_base"].max()

    ax_KH.plot(z_norm, d["K"], color=c, linewidth=1.8,                          label=f"{name} — K")
    ax_KH.plot(z_norm, d["H"], color=c, linewidth=1.8, linestyle='--', alpha=0.6, label=f"{name} — H")

    ax_kpc.plot(z_norm, d["k1"], color=c, linewidth=1.8,                          label=f"{name} — k1")
    ax_kpc.plot(z_norm, d["k2"], color=c, linewidth=1.8, linestyle='--', alpha=0.6, label=f"{name} — k2")

ax_KH.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax_KH.set_xlabel("normalized height (0 = base, 1 = tip)", fontsize=8)
ax_KH.set_ylabel("curvature (1/µm or 1/µm²)", fontsize=8)
ax_KH.set_title("Gaussian K (solid) & Mean H (dashed)", fontsize=9)
ax_KH.tick_params(labelsize=7)
ax_KH.legend(fontsize=6.5)
ax_KH.grid(True, linestyle='--', alpha=0.3)

ax_kpc.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax_kpc.set_xlabel("normalized height (0 = base, 1 = tip)", fontsize=8)
ax_kpc.set_ylabel("principal curvature (1/µm)", fontsize=8)
ax_kpc.set_title("Principal k1 (solid) & k2 (dashed)", fontsize=9)
ax_kpc.tick_params(labelsize=7)
ax_kpc.legend(fontsize=6.5)
ax_kpc.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("hyperbolic_cone_curvature_comparison_overlaid.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: hyperbolic_cone_curvature_comparison_overlaid.png")

# ==============================
# HELPER: per-geometry subplot figure
# ==============================

def plot_per_geometry(geo_list, title, filename):
    n = len(geo_list)
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 2.6))
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.subplots_adjust(hspace=0.45)

    for row_idx, name in enumerate(geo_list):
        d = results[name]
        z = d["z_base"]

        # --- Left: K and H dual axis ---
        ax1  = axes[row_idx, 0]
        ax1r = ax1.twinx()

        ax1.plot(z,  d["K"], color="steelblue",  linewidth=1.8, label="K (Gaussian)")
        ax1r.plot(z, d["H"], color="darkorange",  linewidth=1.8, linestyle='--', label="H (Mean)")

        ax1.set_ylabel("K (1/µm²)",  color="steelblue",  fontsize=8)
        ax1r.set_ylabel("H (1/µm)", color="darkorange", fontsize=8)
        ax1.tick_params(axis='y',  labelcolor="steelblue",  labelsize=7)
        ax1r.tick_params(axis='y', labelcolor="darkorange", labelsize=7)
        ax1.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        ax1.set_xlabel("height from base (µm)", fontsize=8)
        ax1.tick_params(axis='x', labelsize=7)
        ax1.set_title(f"{name}  —  K & H", fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

        # --- Right: Principal curvatures ---
        ax2 = axes[row_idx, 1]
        ax2.plot(z, d["k1"], color="steelblue",  linewidth=1.8, label="k1 (principal max)")
        ax2.plot(z, d["k2"], color="darkorange",  linewidth=1.8, linestyle='--', label="k2 (principal min)")
        ax2.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        ax2.set_xlabel("height from base (µm)", fontsize=8)
        ax2.set_ylabel("principal curvature (1/µm)", fontsize=8)
        ax2.set_title(f"{name}  —  Principal Curvatures", fontsize=9)
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=7)
        ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# ==============================
# FIGURE 2: Ratio geometries (1:1, 2:1, 4:1)
# ==============================

plot_per_geometry(
    RATIO_GEOS,
    "Hyperbolic Cone Curvature Comparison — Ratio Geometries",
    "hyperbolic_cone_curvature_comparison_ratio_geometries.png"
)

# ==============================
# FIGURE 3: Scaled geometries (2:1 scaled 2x, 2:1 scaled 4x)
# ==============================

plot_per_geometry(
    SCALED_GEOS,
    "Hyperbolic Cone Curvature Comparison — Scaled Geometries",
    "hyperbolic_cone_curvature_comparison_scaled_geometries.png"
)

# -----------------------------
# SAVE EXCEL
# -----------------------------

with pd.ExcelWriter("hyperbolic_cone_curvature_comparison.xlsx", engine="openpyxl") as writer:
    for name, d in results.items():
        df = pd.DataFrame({
            "z_base": d["z_base"],
            "K": d["K"],
            "H": d["H"],
            "k1": d["k1"],
            "k2": d["k2"],
            "dK_dz": d["dK_dz"],
        })
        sheet_name = name.replace(":", "-").replace(" ", "_")[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Saved: hyperbolic_cone_curvature_comparison.xlsx")