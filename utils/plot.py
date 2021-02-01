import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from math import radians


def plot_single_trajectory(trajectory, title=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(trajectory[0])
    axes[1].plot(trajectory[1])
    axes[0].set_xlabel("Time")
    axes[1].set_xlabel("Time")
    axes[0].set_ylabel("Theta")
    axes[1].set_ylabel("Radius")
    if not title is None:
        axes[0].set_title(title)


def plot_trajectories(trajectories, title=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for t in trajectories:
        axes[0].plot(t[0])
        axes[1].plot(t[1])
    axes[0].set_xlabel("Time")
    axes[1].set_xlabel("Radius")
    axes[0].set_ylabel("Time")
    axes[1].set_ylabel("Radius")
    if not title is None:
        axes[0].set_title(title)


def make_polar_plot():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="polar")
    ax.set_yticks([])
    #     ax.set_xlim([np.pi+np.pi/7,-np.pi/7])
    ax.set_ylim([0, 420])
    ax.set_xticks([0, radians(10), radians(170), np.pi])
    ax.set_xticklabels(
        ["$0^{\circ}$", "$10^{\circ}$", "$170^{\circ}$", "$180^{\circ}$"],
        FontSize=18)
    ax.spines['polar'].set_visible(False)
    return fig, ax


def make_polar_plot():
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(projection="polar")
    ax.set_yticks([])
    #     ax.set_xlim([np.pi+np.pi/7,-np.pi/7])
    ax.set_ylim([0, 420])
    ax.set_xticks([0, np.pi])
    ax.set_xticklabels(["$0^{\circ}$", "$180^{\circ}$"], FontSize=20)
    ax.spines['polar'].set_visible(False)
    return fig, ax


def make_color_vector(map_type, n_colors):
    return [map_type(x) for x in np.linspace(0, 1, n_colors)]


def make_ridgeplot(curve_matrix,
                   title=None,
                   nbins=20,
                   xlims=[100, 400],
                   xlabel=None,
                   ax=None,
                   cbax=None,
                   cmap=None,
                   fig=None):
    n_timepoints = curve_matrix.shape[1]
    if fig == None:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = fig
    if ax == None:
        spec = matplotlib.gridspec.GridSpec(ncols=2,
                                            nrows=1,
                                            width_ratios=[20, 1])
        ax = fig.add_subplot(spec[0])
        cbax = fig.add_subplot(spec[1])
        cbax.set_ylabel('time')
        cmap = matplotlib.cm.get_cmap('cividis')
        # norm = matplotlib.colors.Normalize(vmin=1, vmax=n_timepoints)

    else:
        ax = ax
        cbax = cbax
        cmap = cmap

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(xlims)

    val_offsets = [i * 0.005 for i in range(n_timepoints)
                   ]  # how vertically spaced lines are
    time_offsets = [0 for i in range(n_timepoints)
                    ]  # how spaced distros are in time axis

    for i in range(n_timepoints):
        vals, bins = np.histogram(curve_matrix[:, i], density=True, bins=nbins)
        midpoints = [(b + bins[bi + 1]) / 2 for bi, b in enumerate(bins[:-1])]
        ax.plot([m + time_offsets[i] for m in midpoints],
                [v + val_offsets[i] for v in vals],
                color=cmap(i / n_timepoints),
                alpha=0.7,
                zorder=n_timepoints - i)

    return fig, ax


def make_distro_ridgeplot(x, curve_matrix, title=None):
    n_timepoints = curve_matrix.shape[1]
    # n_trials = curve_matrix.shape[0]
    fig = plt.figure(figsize=(8, 6))
    spec = matplotlib.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[20, 1])
    ax = fig.add_subplot(spec[0])
    cbax = fig.add_subplot(spec[1])
    cmap = matplotlib.cm.get_cmap('cividis')
    # norm = matplotlib.colors.Normalize(vmin=1, vmax=n_timepoints)
    # cb = matplotlib.colorbar.ColorbarBase(cbax,
    #                                       cmap=cmap,
    #                                       norm=norm,
    #                                       orientation='vertical')
    cbax.set_ylabel('time')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("radius")
    if title:
        ax.set_title(title)

    val_offsets = [i * 0.0005 for i in range(n_timepoints)
                   ]  # how vertically spaced lines are
    time_offsets = [0 for i in range(n_timepoints)
                    ]  # how spaced distros are in time axis

    for i in range(n_timepoints):
        # range of the distribution vs. distribution values
        # matrix should be distribution, num_distributions
        ax.plot([m + time_offsets[i] for m in x],
                [v + val_offsets[i] for v in curve_matrix[:, i]],
                color=cmap(i / n_timepoints),
                alpha=0.7,
                zorder=n_timepoints - i)
    return fig, ax
