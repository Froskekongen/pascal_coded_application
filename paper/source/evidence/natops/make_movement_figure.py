"""A first-reading illustration of the exact four-increment example."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, (schedule, height) = plt.subplots(
    1, 2, figsize=(4.65, 1.33), gridspec_kw={'width_ratios': [1.5, 1]},
    layout='constrained')
increments = [[1, 0, 0, 1], [0, 1, 1, 0]]
colors = ['#28618c', '#be6630']
labels = ['Spread', 'Central']
for row, (z, color, label, style) in enumerate(
        zip(increments, colors, labels, ['o-', 's--'])):
    y = 1 - row
    schedule.text(-.10, y + .35, label, ha='right', va='center',
                  color=color, fontsize=7.5)
    for col, movement in enumerate(z):
        schedule.add_patch(Rectangle((col, y + .02), .9, .66,
                           facecolor=color if movement else '#f4f4f4',
                           edgecolor=color if movement else '#d5d5d5',
                           linewidth=.7, alpha=.13 if movement else 1))
        if movement:
            schedule.annotate('', xy=(col + .45, y + .56),
                              xytext=(col + .45, y + .13),
                              arrowprops={'arrowstyle': '-|>', 'color': color,
                                          'linewidth': 1.5})
        else:
            schedule.text(col + .45, y + .35, 'pause', ha='center',
                          va='center', fontsize=6.8, color='#555555')
    height.plot(np.arange(5), np.r_[0, np.cumsum(z)], style,
                color=color, markersize=2.7, linewidth=1.15)
for col in range(4):
    schedule.text(col + .45, 1.86, str(col + 1), ha='center',
                  va='center', fontsize=7)
schedule.text(-.10, 1.86, 'Interval', ha='right', va='center', fontsize=7)
schedule.set(xlim=(-1.13, 4), ylim=(-.08, 2.13))
schedule.axis('off')
height.set(xlabel='Elapsed intervals', ylabel='Hand height',
           xticks=[0, 1, 2, 3, 4], yticks=[0, 1, 2], ylim=(-.06, 2.1))
height.spines[['top', 'right']].set_visible(False)
height.tick_params(length=2, pad=2, labelsize=7)
height.grid(alpha=.13, linewidth=.5)
fig.savefig(ROOT / 'movement_schedules.pdf', bbox_inches='tight', pad_inches=.02)
