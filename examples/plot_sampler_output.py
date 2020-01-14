import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import corner

from scipy import stats

sns.set_style('darkgrid')


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--burn-in", type=int, default=100)
    return p





if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    df = pd.read_csv(args.input);
    for k in df.columns:
        df[k] = df[k].astype('f')
    df = df.iloc[np.all(np.isfinite(df.values), axis=1)]

    samples = [v for k, v in df.groupby('iteration')]
    samples = samples[args.burn_in:]
    samples = samples[::args.thin]

    columns = np.setdiff1d(samples[0].columns, ['iteration', 'log_probability', 'ensemble_index'])

    data = np.vstack([s[columns].values.astype('f') for s in samples])
    log_p = np.concatenate([s["log_probability"].values for s in samples])
    log_p = np.ma.masked_array(log_p, mask=~np.isfinite(log_p))

    iteration = np.concatenate([s["iteration"].values for s in samples])
    good = log_p >= stats.mstats.mquantiles(log_p, [0.1])
    limits = stats.mstats.mquantiles(data[good], [0.05, 0.95], axis=0).data.T

    def rescale(lim):
        d = 0.05 * (lim[1] - lim[0])
        if d == 0.:
            d = 1e-4
        return (lim[0] - d, lim[1] + d)

    limits = [rescale(lim) for lim in limits]

    bins = [np.linspace(lim[0], lim[1], 20) for lim in limits]

    k = data.shape[1]
    fig = plt.figure(figsize=(16, 16))

    if k > 1:
        grid = plt.GridSpec(k, k, hspace=0.2, wspace=0.2)
    else:
        grid = plt.GridSpec(1, 2, hspace=0.2, wspace=0.2)

    for i in range(len(columns)):
        for j in range(i + 1):
            if i == j or i == len(columns) - 1:
                ax_kwdargs = {}
            else:
                ax_kwdargs = {"xticks": []}

            ax = fig.add_subplot(grid[i, j], **ax_kwdargs)

            if i == j:
                ax.hist(data[:, i], bins=40, color='steelblue')
            else:
                ax.hist2d(data[:, j], data[:, i], bins=[bins[j], bins[i]], norm=mcolors.PowerNorm(0.5))
                ax.set_ylim(limits[i])
                ax.set_xlim(limits[j])
            if i == k - 1:
                ax.set_xlabel(columns[j])
            if j == 0:
                ax.set_ylabel(columns[i])

    if k > 1:
        mid = int(np.ceil(k / 3))
        ax = fig.add_subplot(grid[:mid, (mid + 1):], xticklabels=[])
    else:
        ax = fig.add_subplot(grid[0, 1], xticklabels=[])

    ax.plot(iteration, log_p, '.')
    ylim = stats.mstats.mquantiles(log_p, [0.01, 1.])
    ylim_buffer = 0.05 * (ylim[1] - ylim[0])
    ylim[0] -= ylim_buffer
    ylim[1] += ylim_buffer
    ax.set_ylim(ylim)  
    ax.set_ylabel("log probability")
    ax.set_xlabel("iterations")  
    x_ticks = np.linspace(np.min(iteration), np.max(iteration), 11)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(map(lambda x : str(int(x)), x_ticks))
    plt.show()

