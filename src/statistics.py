"""
Population statistics: cumulative counts and host-to-host scatter.
"""

import numpy as np
import pandas as pd

__all__ = ["cumulative_counts", "per_host_counts", "percentile_band",
           "radial_profile", "stacked_profile", "jackknife_profile",
           "scatter_budget"]


def cumulative_counts(values, edges, direction="above"):
    """
    ``N(> edge)`` for each edge. Ignores NaN (i.e. dark subhalos).

    ``direction="below"`` counts ``N(< edge)`` instead. That is not a stylistic
    option: magnitudes run backwards, so the satellite *luminosity* function
    N(< M_V) is the same statistic as N(> Mstar) and has to be counted the
    other way round.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    order = np.sort(values)
    if direction == "below":
        # 'left' counts entries < edge, so the comparison stays strict.
        return np.searchsorted(order, edges, side="left")
    if direction != "above":
        raise ValueError("direction must be 'above' or 'below'")
    # searchsorted with 'right' counts entries <= edge; N(>edge) is the rest.
    return len(order) - np.searchsorted(order, edges, side="right")


def per_host_counts(df, edges, value_col="mpeak", host_ids=None,
                    host_col="host_id", direction="above"):
    """
    ``N(> edge)`` for every host separately -> array of shape ``(n_host, n_edge)``.

    ``host_ids`` must list *every* host in the sample, including those with no
    surviving rows in ``df``. A host with zero satellites above the threshold
    contributes no rows, so inferring the host list from ``df`` alone silently
    drops it and biases the median and the percentile band high -- which matters
    exactly where the science is, at the bright end where counts are small.
    """
    edges = np.asarray(edges, dtype=float)
    if host_ids is None:
        host_ids = np.unique(df[host_col])
    host_ids = np.asarray(host_ids)

    index = pd.Index(host_ids, name=host_col)
    counts = np.zeros((len(host_ids), len(edges)), dtype=float)

    for hid, group in df.groupby(host_col, sort=False):
        loc = index.get_indexer([hid])[0]
        if loc < 0:  # host outside the requested list
            continue
        counts[loc] = cumulative_counts(group[value_col].to_numpy(), edges,
                                        direction=direction)

    return counts


def percentile_band(counts, percentiles=(16, 50, 84)):
    """Percentiles across hosts (axis 0) of a ``per_host_counts`` array."""
    return np.percentile(np.asarray(counts, dtype=float), percentiles, axis=0)


def radial_profile(df, edges, radius_col="r_proj_over_rvir", host_ids=None,
                   host_col="host_id", normalize=False):
    """
    ``N(< edge)`` in radius, per host -> array of shape ``(n_host, n_edge)``.

    ``normalize=True`` divides each host by its own total inside the largest
    edge. **Do not use it to measure the shape of the radial distribution**
    unless every host has many satellites: a host with two satellites gives a
    normalized profile that jumps 0 -> 0.5 -> 1, and the median over such hosts
    is strongly biased toward a concentrated profile. Use ``stacked_profile``
    instead, which pools the satellites first. The option is here because
    per-host normalization is the right thing for a *scatter* question ("how
    much does the shape vary between well-populated hosts?").
    """
    counts = per_host_counts(df, edges, value_col=radius_col, host_ids=host_ids,
                             host_col=host_col, direction="below")
    if normalize:
        total = counts[:, -1:]
        counts = np.divide(counts, total, out=np.full_like(counts, np.nan),
                           where=total > 0)
    return counts


def jackknife_profile(df, edges, radius_col="r_proj_over_rvir", host_col="host_id",
                      host_ids=None, min_sat=1):
    """
    Mean normalized radial shape ``N(<r)/N(<r_max)`` with a leave-one-out
    jackknife error on the mean.

    Each host is first normalized to its own count inside the outermost edge,
    then the normalized profiles are **averaged** over hosts, one vote per host.
    The uncertainty on that average is a delete-one jackknife over hosts:
    recompute the mean ``N`` times, each time dropping one host, and take

        sigma(r)^2 = (N - 1) / N * sum_j ( p_bar_{(j)}(r) - p_bar(r) )^2 .

    Returns ``(profile, err, n_used)``: ``profile`` and ``err`` have length
    ``len(edges)``, ``n_used`` is the number of hosts kept.

    Normalizing per host before averaging answers a *shape* question and keeps
    a well-populated host and a sparse one on the same footing. The price is a
    bias toward concentrated profiles when hosts have very few satellites (the
    same effect ``radial_profile`` warns about): ``min_sat`` guards against it
    by dropping hosts with fewer than that many satellites inside the outermost
    edge. ``stacked_profile`` makes the opposite choice -- pool first, then
    bootstrap -- and the two agree once every host is well populated.
    """
    per_host = radial_profile(df, edges, radius_col=radius_col, host_ids=host_ids,
                              host_col=host_col, normalize=False)
    total = per_host[:, -1]
    keep = total >= max(int(min_sat), 1)
    if keep.sum() < 2:
        raise ValueError("need at least two hosts with >= min_sat satellites "
                         "for a jackknife")

    p = per_host[keep] / total[keep, None]          # (n, n_edge), each row ends at 1
    n = len(p)
    profile = p.mean(axis=0)
    loo = (n * profile[None, :] - p) / (n - 1)       # delete-one means
    err = np.sqrt((n - 1) / n * np.sum((loo - profile) ** 2, axis=0))
    return profile, err, n


def scatter_budget(counts):
    """
    Split the host-to-host scatter in a count into shot noise and the rest.

    Given ``counts`` for many hosts at one threshold, returns
    ``(mean, sigma_total, sigma_poisson, sigma_intrinsic)`` with

        sigma_intrinsic^2 = sigma_total^2 - <N>,

    since a pure Poisson process would give ``sigma^2 = <N>``. The residual is
    real halo-to-halo variance in the number of subhalos accreted -- the part
    that does *not* average away if you observe one host very well. Clipped at
    zero, because the estimate is noisy and can go slightly negative when the
    intrinsic term is small.
    """
    counts = np.asarray(counts, dtype=float)
    mean = counts.mean()
    sigma = counts.std(ddof=1)
    poisson = np.sqrt(mean)
    intrinsic = np.sqrt(max(sigma ** 2 - mean, 0.0))
    return mean, sigma, poisson, intrinsic


def stacked_profile(df, edges, radius_col="r_proj_over_rvir", host_col="host_id",
                    host_ids=None, n_boot=200, rng=None):
    """
    Normalized radial shape ``N(<r)/N(<r_max)`` from all satellites **pooled**,
    with a 16-84% band from bootstrapping over hosts.

    Returns ``(profile, lo, hi)``, each of length ``len(edges)``.

    Pooling first and normalizing once is what a stacked measurement of a real
    survey does, and unlike a median of per-host normalized profiles it is
    unbiased when hosts have few satellites. The bootstrap resamples *hosts*,
    not satellites, because satellites of the same host are not independent.
    """
    rng = np.random.default_rng(rng)
    edges = np.asarray(edges, dtype=float)

    if host_ids is None:
        host_ids = np.unique(df[host_col])
    host_ids = np.asarray(host_ids)

    by_host = {h: g[radius_col].to_numpy() for h, g in df.groupby(host_col, sort=False)}
    empty = np.array([])

    def _profile(ids):
        vals = np.concatenate([by_host.get(h, empty) for h in ids]) if len(ids) else empty
        c = cumulative_counts(vals, edges, direction="below").astype(float)
        return c / c[-1] if c[-1] > 0 else np.full(len(edges), np.nan)

    profile = _profile(host_ids)
    boots = np.array([_profile(rng.choice(host_ids, size=len(host_ids), replace=True))
                      for _ in range(int(n_boot))])
    lo, hi = np.nanpercentile(boots, [16, 84], axis=0)
    return profile, lo, hi
