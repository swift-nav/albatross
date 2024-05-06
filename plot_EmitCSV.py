import pandas as pd
import matplotlib.pyplot as plt

MARKERCOLORS = ['black', 'blue']
GROUPSIZE = 75
GROUPCOLORS = ["skyblue", "lightcoral", "springgreen", "navajowhite"]

preds = pd.read_csv("example.csv")
points = pd.read_csv("example_points.csv")


fig, ax = plt.subplots(2, figsize=(8,6))
for group in preds['group']:
    xlow = group * GROUPSIZE
    xhigh = xlow + GROUPSIZE
    xfill = [xlow, xhigh, xhigh, xlow]
    yfill = [-0.5, -0.5, 1.5, 1.5]
    ax[0].fill(xfill, yfill, color=GROUPCOLORS[group], alpha=0.9)
for label, df in preds.groupby('type'):
    df.plot(x='x', y='mean', label=label, ax=ax[0])
iColor = 0
for label, df in points.groupby('type'):
    df.plot.scatter(x='x', y='y', c=MARKERCOLORS[iColor], ax=ax[0], label=label)
    iColor += 1
plt.legend()

for group in preds['group']:
    xlow = group * GROUPSIZE
    xhigh = xlow + GROUPSIZE
    xfill = [xlow, xhigh, xhigh, xlow]
    yfill = [0, 0, 0.05, 0.05]
    ax[1].fill(xfill, yfill, color=GROUPCOLORS[group], alpha=0.9)
for label, df in preds.groupby('type'):
    df.plot(x='x', y='marginal', label=label, ax=ax[1])

plt.show()

