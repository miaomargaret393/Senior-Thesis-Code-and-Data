import matplotlib.pyplot as plt
import numpy as np

geometries = ['Hemi-ellipsoid', 'Cone', 'Hyperbolic', 'Triangular Prism', 'Square Pyramid']
ratios = ['1:1', '2:1', '4:1']

surface_area = [
    [780.796, 1315.707, 2413.066],
    [571.860, 905.327, 1601.433],
    [458.413, 571.777, 822.091],
    [953.115, 1602.699, 2939.008],
    [728.115, 1152.699, 2039.008],
]

volume = [
    [1767.146, 3534.292, 7068.583],
    [883.573, 1767.146, 3534.292],
    [405.241, 537.16, 869.328],
    [1687.500, 3375.00, 6750.00],
    [1125, 2250, 4500],
]

vol_sa = [
    [2.263, 2.686, 2.930],
    [1.545, 1.952, 2.207],
    [0.884, 0.939, 1.057],
    [1.770, 2.106, 2.297],
    [1.545, 1.951, 2.207],
]

x = np.arange(len(geometries))
width = 0.25
colors = ['#4C72B0', '#DD8452', '#55A868']

fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.suptitle('Geometry Metrics by Aspect Ratio', fontsize=16, fontweight='bold', y=0.98)

datasets = [
    (surface_area, 'Surface Area (μm²)', 'Surface Area'),
    (volume,       'Volume (μm³)',        'Volume'),
    (vol_sa,       'Volume / Surface Area', 'Vol/SA Ratio'),
]

for ax, (data, ylabel, title) in zip(axes, datasets):
    for i, (ratio, color) in enumerate(zip(ratios, colors)):
        vals = [row[i] for row in data]
        bars = ax.bar(x + i * width, vals, width, label=ratio, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f'{val:.0f}' if val > 10 else f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7.5, color='#333')

    ax.set_title(title, fontsize=13, pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels(geometries, fontsize=10)
    ax.legend(title='Aspect Ratio', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('geometry_metrics.png', dpi=150, bbox_inches='tight')
plt.show()