import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -------------------------
# SETUP
# -------------------------
geometries = ["Hemi-Ellipsoid", "Cone", "Hyperbolic Cone", "Triangular Prism", "Square Pyramid"]
x = np.arange(len(geometries))
width = 0.2
z = np.array([0.0, 0.6, 1.0])

colors = {
    "Hemi-Ellipsoid": "#1f77b4",
    "Cone": "#ff7f0e",
    "Hyperbolic Cone": "#2ca02c",
    "Triangular Prism": "#9467bd",
    "Square Pyramid": "#8c564b"
}
color_list = [colors[g] for g in geometries]

# -------------------------
# DATA
# -------------------------

# ---- 2:1 ----
base_20_2 = [6.93, 7.03, 6.37, 6.42, 6.83]
trans_20_2 = [6.63, 6.12, 6.44, 7.25, 5.73]
tip_20_2 = [3.33, 1.52, 1.34, 3.76, 2.08]

base_44_2 = [3.21, 5.75, 6.50, 6.98, 6.98]
trans_44_2 = [5.82, 5.29, 5.34, 6.49, 5.64]
tip_44_2 = [2.00, 1.65, 1.03, 1.45, 1.45]

nuc_height_20_2 = [93.17, 48.94, 38.98, 69.82, 65.58]
nuc_height_44_2 = [88.69, 54.09, 35.64, 52.12, 48.69]

nuc_base_20_2 = [4.91, 2.36, 1.67, 4.08, 3.06]
nuc_base_44_2 = [10.61, 3.04, 1.25, 2.63, 2.63]

reach_20_2 = [4/5, 3/5, 0/5, 0/5, 0/5]
reach_44_2 = [5/5, 5/5, 2/5, 2/5, 0/5]

# ---- 1:1 ----
base_20_1 = [8.50, 6.89, 8.52, 6.98, 7.78]
trans_20_1 = [7.21, 6.53, 7.10, 7.35, 6.70]
tip_20_1 = [2.31, 2.20, 1.45, 2.65, 1.57]

base_44_1 = [6.85, 6.98, 7.16, 6.68, 6.07]
trans_44_1 = [2.36, 7.60, 6.04, 6.49, 5.21]
tip_44_1 = [1.06, 2.67, 1.50, 2.08, 1.77]

nuc_height_20_1 = [74.55, 70.46, 62.17, 91.78, 42.72]
nuc_height_44_1 = [69.82, 42.26, 38.67, 58.21, 48.69]

nuc_base_20_1 = [1.94, 1.75, 1.15, 2.90, 0.97]
nuc_base_44_1 = [1.70, 0.93, 0.69, 1.49, 1.26]

reach_20_1 = [5/5, 5/5, 3/5, 3/5, 0/5]
reach_44_1 = [5/5, 3/5, 4/5, 4/5, 1/5]

# -------------------------
# FIGURE 1: THICKNESS PANELS
# -------------------------
fig1, axs = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

for i, g in enumerate(geometries):
    ax = axs[i]
    c = colors[g]

    # 2:1
    ax.plot(z, [base_20_2[i], trans_20_2[i], tip_20_2[i]], color=c, marker='o')
    ax.plot(z, [base_44_2[i], trans_44_2[i], tip_44_2[i]], color=c, linestyle='--', marker='o')

    # 1:1
    ax.plot(z, [base_20_1[i], trans_20_1[i], tip_20_1[i]], color=c, marker='^')
    ax.plot(z, [base_44_1[i], trans_44_1[i], tip_44_1[i]], color=c, linestyle='--', marker='^')

    ax.set_title(g, fontsize=10)
    ax.set_xticks([0, 0.6, 1])
    ax.set_xticklabels(["Base", "Trans", "Tip"])
    ax.grid(alpha=0.2)

    # legend inside each subplot
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='20 hr'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='44 hr'),
        Line2D([0], [0], color='black', marker='o', lw=0, label='2:1'),
        Line2D([0], [0], color='black', marker='^', lw=0, label='1:1'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, frameon=False)

axs[0].set_ylabel("Thickness (µm)")

plt.tight_layout()
plt.savefig("Figure1_thickness.png", dpi=300)
plt.show()

# -------------------------
# FIGURE 2: BAR PLOTS
# -------------------------
fig2, axs = plt.subplots(1, 3, figsize=(14, 4))

# (A) Nuclear climbing
axs[0].bar(x - width, nuc_height_20_2, width, color=color_list)
axs[0].bar(x, nuc_height_44_2, width, color=color_list, hatch='//')
axs[0].bar(x + width, nuc_height_20_1, width, color=color_list, alpha=0.5)
axs[0].bar(x + 2*width, nuc_height_44_1, width, color=color_list, hatch='//', alpha=0.5)

axs[0].set_title("Nuclear climbing")
axs[0].set_ylabel("% height")
axs[0].set_xticks(x)
axs[0].set_xticklabels(geometries, rotation=30)

# legend (only once)
legend_elements = [
    Patch(facecolor='gray', label='2:1'),
    Patch(facecolor='gray', alpha=0.5, label='1:1'),
    Patch(facecolor='gray', hatch='//', label='44 hr'),
    Patch(facecolor='gray', label='20 hr'),
]
axs[0].legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=False)

# (B) Nuc/Base
axs[1].bar(x - width, nuc_base_20_2, width, color=color_list)
axs[1].bar(x, nuc_base_44_2, width, color=color_list, hatch='//')
axs[1].bar(x + width, nuc_base_20_1, width, color=color_list, alpha=0.5)
axs[1].bar(x + 2*width, nuc_base_44_1, width, color=color_list, hatch='//', alpha=0.5)

axs[1].set_title("Nuc/Base")
axs[1].set_xticks(x)
axs[1].set_xticklabels(geometries, rotation=30)

# (C) Traversal
axs[2].bar(x - width, reach_20_2, width, color=color_list)
axs[2].bar(x, reach_44_2, width, color=color_list, hatch='//')
axs[2].bar(x + width, reach_20_1, width, color=color_list, alpha=0.5)
axs[2].bar(x + 2*width, reach_44_1, width, color=color_list, hatch='//', alpha=0.5)

axs[2].set_title("Traversal success")
axs[2].set_ylabel("Fraction")
axs[2].set_xticks(x)
axs[2].set_xticklabels(geometries, rotation=30)

plt.tight_layout()
plt.savefig("Figure2_bars.png", dpi=300)
plt.show()