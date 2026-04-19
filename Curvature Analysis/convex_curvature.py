import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------
# Semi-oval (hemi-ellipse) surface of revolution
# r(z) = a * sqrt(1 - (z/b)^2),  0 <= z <= b
#
# Geometries:
#   1:1  — hemisphere:        b=30 (since it will always be elipsoid and the height out of the substrate is 15),  a=15
#   2:1  — tall ellipsoid:    b=60,  a=15
#   2:1 scaled 2x:            b=120,  a=30
#   2:1 scaled 4x:            b=240, a= 60  (NEW - but wait, spec says a=30, b=120)
#   4:1:                      b=120,  a=15
#   6:1:                      b=240, a=15
# -------------------------------------------------

# -----------------------------
# GEOMETRY DEFINITIONS
# -----------------------------

geometries = {
    "1:1":              {"a": 15.0, "b": 30.0},
    "2:1":              {"a": 15.0, "b": 60.0},
    "2:1 scaled 2x":    {"a": 30.0, "b": 120.0},
    "2:1 scaled 4x":    {"a": 60.0, "b": 240.0},
    "4:1":              {"a": 15.0, "b": 120.0},
    "6:1":              {"a": 15.0, "b": 240.0},
}

# Groups for per-geometry figures
RATIO_GEOS  = ["1:1", "2:1", "4:1"]
SCALED_GEOS = ["2:1 scaled 2x", "2:1 scaled 4x"]

# -----------------------------
# SHARED PARAMETERS
# -----------------------------

r_min_cutoff = 0.5
N = 4000

COLORS = {
    "1:1":              "steelblue",
    "2:1":              "darkorange",
    "2:1 scaled 2x":    "seagreen",
    "2:1 scaled 4x":    "teal",
    "4:1":              "mediumpurple",
    "6:1":              "crimson",
}

# -----------------------------
# COMPUTE CURVATURES
# -----------------------------

def compute_ellipsoid_curvature(a, b, N=4000, r_min_cutoff=0.5):
    z = np.linspace(0.0, b, N)

    inside = np.clip(1.0 - (z / b)**2, 0.0, None)
    r = a * np.sqrt(inside)

    dr_dz   = np.gradient(r, z)
    d2r_dz2 = np.gradient(dr_dz, z)

    # Gaussian curvature for surface of revolution
    K = -d2r_dz2 / (r * (1.0 + dr_dz**2)**2)

    # Meridional (k_mer) and circumferential (k_circ) principal curvatures
    k_mer  = -d2r_dz2 / (1.0 + dr_dz**2)**1.5
    k_circ =  1.0 / (r * np.sqrt(1.0 + dr_dz**2))

    # Remove singular tip where r -> 0
    mask = r >= r_min_cutoff
    z, r, K = z[mask], r[mask], K[mask]
    k_mer, k_circ = k_mer[mask], k_circ[mask]
    dr_dz = dr_dz[mask]

    k1 = np.maximum(k_mer, k_circ)
    k2 = np.minimum(k_mer, k_circ)
    H  = (k1 + k2) / 2.0
    dK_dz = np.gradient(K, z)

    # height from base = z directly (base is z=0 for ellipsoids)
    z_base = z.copy()

    return {"z_base": z_base, "r": r, "K": K, "H": H,
            "k1": k1, "k2": k2, "dK_dz": dK_dz}

results = {}
for name, g in geometries.items():
    results[name] = compute_ellipsoid_curvature(g["a"], g["b"])

# -----------------------------
# SUMMARY TABLE
# -----------------------------

print("\n==== ELLIPSOID CURVATURE SUMMARY TABLE ====\n")
print(f"{'Geometry':<22} {'K_min':>12} {'K_max':>12} {'H_max':>12} {'H_min':>12} {'k1_max':>12} {'k2_min':>12}")
print("-" * 96)
for name, d in results.items():
    print(f"{name:<22} {d['K'].min():>12.4e} {d['K'].max():>12.4e} "
          f"{d['H'].max():>12.4e} {d['H'].min():>12.4e} "
          f"{d['k1'].max():>12.4e} {d['k2'].min():>12.4e}")

# ==============================
# FIGURE 1: OVERLAID COMPARISONS (all geometries)
# Sized for 8.5 x 11 page (portrait)
# ==============================

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
fig.suptitle("Ellipsoid Curvature Comparison", fontsize=11, fontweight='bold')

ax_KH  = axes[0]
ax_kpc = axes[1]

for name, d in results.items():
    c      = COLORS[name]
    z_norm = d["z_base"] / d["z_base"].max()

    ax_KH.plot(z_norm, d["K"], color=c, linewidth=1.8,                           label=f"{name} — K")
    ax_KH.plot(z_norm, d["H"], color=c, linewidth=1.8, linestyle='--', alpha=0.6, label=f"{name} — H")

    ax_kpc.plot(z_norm, d["k1"], color=c, linewidth=1.8,                           label=f"{name} — k1")
    ax_kpc.plot(z_norm, d["k2"], color=c, linewidth=1.8, linestyle='--', alpha=0.6, label=f"{name} — k2")

ax_KH.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax_KH.set_xlabel("normalized height (0 = base, 1 = apex)", fontsize=8)
ax_KH.set_ylabel("curvature (1/µm or 1/µm²)", fontsize=8)
ax_KH.set_title("Gaussian K (solid) & Mean H (dashed)", fontsize=9)
ax_KH.tick_params(labelsize=7)
ax_KH.legend(fontsize=6.5)
ax_KH.grid(True, linestyle='--', alpha=0.3)

ax_kpc.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax_kpc.set_xlabel("normalized height (0 = base, 1 = apex)", fontsize=8)
ax_kpc.set_ylabel("principal curvature (1/µm)", fontsize=8)
ax_kpc.set_title("Principal k1 (solid) & k2 (dashed)", fontsize=9)
ax_kpc.tick_params(labelsize=7)
ax_kpc.legend(fontsize=6.5)
ax_kpc.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("ellipsoid_curvature_comparison_overlaid.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ellipsoid_curvature_comparison_overlaid.png")

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

        ax1.plot(z,  d["K"], color="steelblue",  linewidth=1.8,              label="K (Gaussian)")
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
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

        # --- Right: Principal curvatures ---
        ax2 = axes[row_idx, 1]
        ax2.plot(z, d["k1"], color="steelblue",  linewidth=1.8,              label="k1 (principal max)")
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
    "Ellipsoid Curvature Comparison — Ratio Geometries",
    "ellipsoid_curvature_comparison_ratio_geometries.png"
)

# ==============================
# FIGURE 3: Scaled geometries (2:1 scaled 2x, 2:1 scaled 4x)
# ==============================

plot_per_geometry(
    SCALED_GEOS,
    "Ellipsoid Curvature Comparison — Scaled Geometries",
    "ellipsoid_curvature_comparison_scaled_geometries.png"
)

# -----------------------------
# SAVE EXCEL
# -----------------------------

with pd.ExcelWriter("ellipsoid_curvature_comparison.xlsx", engine="openpyxl") as writer:
    for name, d in results.items():
        df = pd.DataFrame({
            "z_base": d["z_base"],
            "r":      d["r"],
            "K":      d["K"],
            "H":      d["H"],
            "k1":     d["k1"],
            "k2":     d["k2"],
            "dK_dz":  d["dK_dz"],
        })
        sheet_name = name.replace(":", "-").replace(" ", "_")[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Saved: ellipsoid_curvature_comparison.xlsx")