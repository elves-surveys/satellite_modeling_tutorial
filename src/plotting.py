"""Plot styling and a couple of helpers used repeatedly in the notebook."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["use_tutorial_style", "plot_band", "sky_to_mollweide",
           "plot_sky_coverage", "HOST_COLORS", "SURVEY_COLORS"]

HOST_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# One colour per census survey, deepest first.
SURVEY_COLORS = {"des_y6": "#C44E52", "delve_dr3": "#DD8452", "ps1_dr1": "#4C72B0"}


def use_tutorial_style():
    """Readable-on-a-projector defaults. Safe to call more than once."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "axes.linewidth": 1.1,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "lines.linewidth": 2.0,
    })


def plot_band(ax, x, counts, color="C0", label=None, percentiles=(16, 50, 84),
              alpha=0.22, show_mean=False, show_median=True,**kwargs):
    """
    Median line plus a percentile band across hosts.

    ``counts`` is the ``(n_host, n_x)`` array from ``statistics.per_host_counts``.
    """
    lo, mid, hi = np.nanpercentile(counts, percentiles, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    if show_median:
        ax.plot(x, mid, color=color, label=label, **kwargs)
    if show_mean:
        ax.plot(x, np.nanmean(counts, axis=0), color=color, label=label, **kwargs)
    return mid


def label_cumulative(ax, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def sky_to_mollweide(ra, dec):
    """
    ``(ra, dec)`` in degrees -> the ``(lon, lat)`` radians a Mollweide axes wants.

    Right ascension is flipped so that it increases to the *left*, which is the
    astronomical convention for looking at the sky from the inside. Forgetting
    the flip mirrors the footprint and nobody notices until they try to find a
    familiar object on it.
    """
    lon = np.radians(180.0 - (np.asarray(ra, dtype=float) % 360.0))
    return lon, np.radians(np.asarray(dec, dtype=float))


def plot_sky_coverage(ax, footprint, colors=None, threshold=0.5, alpha=0.85):
    """
    Fill each survey's footprint on a Mollweide axes.

    ``footprint`` is what ``io.load_census_footprint`` returns; ``ax`` must have
    been created with ``projection="mollweide"``. Cells covered less than
    ``threshold`` are left blank. Returns ``{survey: area in deg^2}``.
    """
    colors = colors or SURVEY_COLORS
    cov = footprint["coverage"]
    lon = np.radians(180.0 - footprint["ra_edges"])
    lat = np.arcsin(footprint["sindec_edges"])
    X, Y = np.meshgrid(lon, lat)
    cell_area = 4 * np.pi * (180 / np.pi) ** 2 / cov[0].size

    areas = {}
    for k, name in enumerate(footprint["surveys"]):
        c = cov[k]
        # antialiased=False matters: with it on, matplotlib blends the seam
        # between every pair of the 64,800 quads and the footprint comes out
        # visibly speckled.
        ax.pcolormesh(X, Y, np.ma.masked_where(c < threshold, np.ones_like(c)),
                      cmap=mpl.colors.ListedColormap([colors[name]]),
                      alpha=alpha, shading="flat", rasterized=True,
                      antialiased=False, linewidth=0)
        areas[name] = c.sum() * cell_area

    ax.set_xticks(np.radians([-150, -100, -50, 0, 50, 100, 150]))
    ax.set_xticklabels([f"{int(round((180 - t) % 360))}$^\\circ$"
                        for t in (-150, -100, -50, 0, 50, 100, 150)], fontsize=8)
    ax.grid(True, color="0.75", lw=0.5, alpha=0.6)
    return areas
