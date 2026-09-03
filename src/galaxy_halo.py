"""
Layer 2 of the forward model: the galaxy-halo connection.

Two conceptually separate questions, kept as two separate functions:

    occupation    P(galaxy | Mpeak)          -- does this subhalo host a galaxy?
    SHMR          P(Mstar | Mpeak, galaxy)   -- if so, how many stars does it have?

Keeping them apart is the point of Section 1. A subhalo that fails the
occupation draw is dark; it does not get a small stellar mass, it gets none.
"""

import numpy as np
from scipy.special import erf

__all__ = [
    "occupation_fraction",
    "assign_occupation",
    "mean_stellar_mass",
    "assign_stellar_mass",
    "populate_subhalos",
    "FIDUCIAL_OCCUPATION",
    "FIDUCIAL_SHMR",
]


# ---------------------------------------------------------------------------
# Fiducial parameters
# ---------------------------------------------------------------------------
# logM50 sits between the two ends of the literature: Nadler+20 fit ~10^7.5 to
# the MW satellites, while the Dooley+16 / Barber+14 reionization curve is a
# much sharper cutoff at 10^8.75. That factor of ~20 disagreement is itself a
# tutorial point -- it is the single most uncertain number in the whole model.
FIDUCIAL_OCCUPATION = dict(logM50=8.0, sigma_gal=0.4)

# A plain power law with these values reproduces Danieli+23 to 0.002 dex and
# Nadler+20 to 0.06 dex over Mpeak = 10^8 - 10^11 (see reference.py). The
# transparent model is not a toy: over the range that matters it *is* the
# literature relation.
FIDUCIAL_SHMR = dict(alpha=2.0, logM0=10.0, logMstar0=6.5, sigma_logMstar=0.2)


# ---------------------------------------------------------------------------
# Section 1: occupation
# ---------------------------------------------------------------------------
def occupation_fraction(mpeak, logM50=8.0, sigma_gal=0.4):
    """
    Probability that a subhalo of peak mass ``mpeak`` hosts a luminous galaxy.

    A smooth step in log mass,

        f_gal = 0.5 * [ 1 + erf( (log Mpeak - log M50) / (sqrt(2) sigma_gal) ) ],

    so ``logM50`` is the mass at which half of all halos light up and
    ``sigma_gal`` is the width of the transition in dex. ``sigma_gal -> 0``
    gives a hard threshold.
    """
    lgm = np.log10(np.asarray(mpeak, dtype=float))
    if sigma_gal <= 0:
        return (lgm > logM50).astype(float)
    return 0.5 * (1.0 + erf((lgm - logM50) / (np.sqrt(2.0) * sigma_gal)))


def assign_occupation(mpeak, logM50=8.0, sigma_gal=0.4, rng=None):
    """Bernoulli draw per subhalo: ``True`` if it hosts a galaxy."""
    rng = np.random.default_rng(rng)
    p = occupation_fraction(mpeak, logM50=logM50, sigma_gal=sigma_gal)
    return rng.uniform(size=np.shape(p)) < p


# ---------------------------------------------------------------------------
# Section 2: stellar-to-halo mass relation
# ---------------------------------------------------------------------------
def mean_stellar_mass(mpeak, alpha=2.0, logM0=10.0, logMstar0=6.5):
    """
    Mean ``log10 Mstar`` from a power-law SHMR, in dex:

        log Mstar = logMstar0 + alpha * (log Mpeak - logM0).

    ``logM0`` is a pivot, not a free parameter -- fix it and fit the other two,
    otherwise ``logMstar0`` and ``alpha`` are degenerate.
    """
    lgm = np.log10(np.asarray(mpeak, dtype=float))
    return logMstar0 + alpha * (lgm - logM0)


def assign_stellar_mass(mpeak, alpha=2.0, logM0=10.0, logMstar0=6.5,
                        sigma_logMstar=0.2, rng=None):
    """
    Draw ``log10 Mstar`` with lognormal scatter about the mean relation.

    The scatter is not cosmetic. The subhalo mass function is steep, so at fixed
    Mstar there are far more halos scattering up than down: adding scatter
    raises the abundance of bright satellites even though the mean relation is
    unchanged. This is Eddington bias, and Section 2 shows it directly.
    """
    rng = np.random.default_rng(rng)
    mu = mean_stellar_mass(mpeak, alpha=alpha, logM0=logM0, logMstar0=logMstar0)
    if sigma_logMstar <= 0:
        return mu
    return rng.normal(loc=mu, scale=sigma_logMstar)


# ---------------------------------------------------------------------------
# Section 3: put the two together
# ---------------------------------------------------------------------------
def populate_subhalos(subs, occupation_params=None, shmr_params=None,
                      rng=None, keep_dark=False):
    """
    Run occupation and the SHMR over a subhalo table.

    Returns a copy with ``is_luminous``, ``f_gal``, ``logmstar`` and ``mstar``
    added. Dark subhalos carry ``NaN`` stellar mass and are dropped unless
    ``keep_dark=True``.

    One ``rng`` drives both draws, so a fixed seed gives a reproducible mock.
    """
    rng = np.random.default_rng(rng)
    occ = dict(FIDUCIAL_OCCUPATION, **(occupation_params or {}))
    shmr = dict(FIDUCIAL_SHMR, **(shmr_params or {}))

    out = subs.copy()
    mpeak = out["mpeak"].to_numpy()

    out["f_gal"] = occupation_fraction(mpeak, **occ)
    out["is_luminous"] = assign_occupation(mpeak, rng=rng, **occ)

    logmstar = assign_stellar_mass(mpeak, rng=rng, **shmr)
    out["logmstar"] = np.where(out["is_luminous"], logmstar, np.nan)
    out["mstar"] = 10.0 ** out["logmstar"]

    return out if keep_dark else out[out["is_luminous"]].copy()
