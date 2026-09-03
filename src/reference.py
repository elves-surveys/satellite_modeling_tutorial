"""
Literature curves for comparison, kept self-contained so the tutorial does not
depend on the ELVES-Dwarf repo.

Each is reduced to the minimum needed to draw a line.
"""

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["nadler20_shmr", "danieli23_shmr", "behroozi19_shmr",
           "dooley16_occupation",
           "SHMR_POWERLAW_FITS", "SHMR_SCATTER", "BEHROOZI19_PARAMS",
           "DOOLEY16_ERF", "DOOLEY16_TABLE",
           "MW_DWARFS", "mw_dwarfs",
           "MW_CENSUS_P50", "MW_CENSUS_FOOTPRINT", "MW_CENSUS_DISTANCE_RANGE",
           "SKY_AREA_DEG2", "TAN25_POPULATION", "census_p50_mv",
           "TAN25_SIZE_LUMINOSITY", "tan25_mean_log_rhalf",
           "CENSUS_WIDTH", "CENSUS_BRIGHT_ALL_SKY_MV",
           "TAN25_BOX", "tan25_luminosity_function", "tan25_lf_quantiles",
           "tan25_radial_profile", "tan25_radial_quantiles"]


# Nadler+20 (arXiv:1912.03303), MW satellites. Columns: log Mpeak, then the
# 16th and 84th percentile of log Mstar.
_N20 = np.array([
    [7.605029, 1.446168, 2.346203],
    [8.193258, 2.392491, 3.373748],
    [8.679399, 3.339453, 4.376960],
    [9.082215, 4.111935, 5.274444],
    [9.438700, 4.822207, 6.047231],
    [10.035949, 5.993474, 6.931010],
    [10.586866, 7.102531, 7.996273],
    [10.989175, 8.312517, 8.974993],
])


# Range over which the Nadler+20 table is defined. np.interp would silently
# clamp to the end values outside it, which draws a flat line that looks like a
# physical turnover but is only an artifact -- so we return NaN instead.
NADLER20_RANGE = (_N20[0, 0], _N20[-1, 0])


def nadler20_shmr(mpeak, scatter=False):
    """
    Median ``log10 Mstar`` (or the 1-sigma scatter) from Nadler+20.

    ``NaN`` outside the tabulated range, log Mpeak = 7.61 - 10.99.
    """
    lgm = np.log10(np.asarray(mpeak, dtype=float))
    col = (_N20[:, 2] - _N20[:, 1]) / 2 if scatter else (_N20[:, 1] + _N20[:, 2]) / 2
    out = np.interp(lgm, _N20[:, 0], col)
    return np.where((lgm >= NADLER20_RANGE[0]) & (lgm <= NADLER20_RANGE[1]), out, np.nan)


def danieli23_shmr(mpeak):
    """
    Median ``log10 Mstar`` from Danieli+23 (arXiv:2210.14233).

    Their relation is a Behroozi+19 functional form with slope alpha = 2.10.
    Over 10^8 - 10^11 that is a straight line to 0.002 dex, so we carry the
    power law rather than the full parameterization.
    """
    lgm = np.log10(np.asarray(mpeak, dtype=float))
    return 6.488 + 2.098 * (lgm - 10.0)


# Power-law (alpha, logMstar0 at a pivot of logM0 = 10) fits over 10^8 - 10^11,
# with the rms residual of the fit in dex.
SHMR_POWERLAW_FITS = {
    "Nadler+20":  dict(alpha=1.977, logMstar0=6.476, logM0=10.0, rms=0.064),
    "Danieli+23": dict(alpha=2.098, logMstar0=6.488, logM0=10.0, rms=0.002),
}


# Behroozi+19 (UniverseMachine, arXiv:1806.07893) Eq. J1-J2, with the "True All
# Sat. Excl." z = 0 row of their Table J1. This is the *intrinsic* relation, not
# the one convolved with observational scatter, and Mpeak is a Bryan & Norman
# virial mass -- the same definition the SatGen catalog uses.
BEHROOZI19_PARAMS = dict(logM1=11.889, eps=-1.432, alpha=1.959, beta=0.464,
                         delta=0.319, log_gamma=-0.812)

# UniverseMachine is calibrated by observed stellar-mass functions and clustering,
# which run out around Mstar ~ 10^8, i.e. Mpeak ~ 10^10.5. Below that the curve is
# an extrapolation of the fitted functional form -- draw it, but say so.
BEHROOZI19_CALIBRATED_ABOVE = 10.5


def behroozi19_shmr(mpeak, alpha=None):
    """
    Median ``log10 Mstar`` from Behroozi+19, their double-power-law form

        log Mstar = log M1 + eps - log10(10^(-alpha x) + 10^(-beta x))
                    + gamma exp(-x^2 / 2 delta^2),      x = log10(Mpeak / M1).

    ``alpha`` overrides the low-mass slope only; ``danieli23_shmr`` is this same
    curve with ``alpha = 2.10``. Below ``BEHROOZI19_CALIBRATED_ABOVE`` this is an
    extrapolation (see that constant).
    """
    p = BEHROOZI19_PARAMS
    a = p["alpha"] if alpha is None else alpha
    x = np.log10(np.asarray(mpeak, dtype=float)) - p["logM1"]
    return (p["logM1"] + p["eps"]
            - np.log10(10.0 ** (-a * x) + 10.0 ** (-p["beta"] * x))
            + 10.0 ** p["log_gamma"] * np.exp(-0.5 * (x / p["delta"]) ** 2))


# Intrinsic scatter in log Mstar at fixed Mpeak, in dex, as each paper reports it.
# These are not measurements of the same thing: Behroozi+19 fit a constant 0.2 dex
# to massive centrals and carry it down; Danieli+23 infer a *small* scatter from
# the ELVES satellite abundances; Nadler+20's is a posterior width from the MW
# satellites and grows toward the faint end. Where a model does not measure the
# scatter at dwarf masses, it is assuming one.
SHMR_SCATTER = {
    "Behroozi+19": 0.20,
    "Danieli+23":  0.06,
}


# Dooley+16 Fig. 12 fiducial curve (from the Barber+14 reionization model):
# fraction of halos hosting a luminous galaxy vs log Mpeak.
_DOOLEY16 = np.array([
    [7.00, 0.00], [7.43, 0.01], [7.74, 0.01], [7.99, 0.04], [8.15, 0.04],
    [8.25, 0.09], [8.33, 0.14], [8.35, 0.15], [8.45, 0.19], [8.55, 0.24],
    [8.65, 0.33], [8.72, 0.43], [8.75, 0.47], [8.77, 0.52], [8.82, 0.60],
    [8.85, 0.66], [8.92, 0.73], [9.05, 0.81], [9.14, 0.86], [9.19, 0.90],
    [9.25, 0.94], [9.35, 0.99], [9.45, 1.00], [9.60, 1.00],
])

# An erf fit to the curve above, i.e. the occupation_fraction() parameters that
# reproduce this reionization model.
DOOLEY16_ERF = dict(logM50=8.752, sigma_gal=0.333)


# The tabulated points themselves, for plotting: (log Mpeak, f_gal).
DOOLEY16_TABLE = _DOOLEY16


def dooley16_occupation(mpeak):
    """Luminous fraction vs Mpeak, interpolated from the Dooley+16 curve."""
    f = interp1d(_DOOLEY16[:, 0], _DOOLEY16[:, 1],
                 bounds_error=False, fill_value=(0.0, 1.0))
    return f(np.log10(np.asarray(mpeak, dtype=float)))


# ---------------------------------------------------------------------------
# A handful of real Milky Way satellites
# ---------------------------------------------------------------------------
# Rounded literature values compiled from McConnachie (2012), Munoz et al.
# (2018) and Simon (2019): name, M_V, half-light radius in pc, heliocentric
# distance in kpc.
#
# READ THIS BEFORE USING IT. This is *not* a complete sample, and it is not
# homogeneous -- it is a familiar subset, chosen to span the full range of
# luminosity and size, for **visually sanity-checking the size and surface-
# brightness models** in notebook 03. Do not count rows and compare the number
# with a mock: that comparison needs a completeness model, which is notebook 05.
_MW_DWARFS = [
    ("LMC",              -18.1, 2115.,  51.),
    ("SMC",              -16.8, 1100.,  64.),
    ("Sagittarius",      -13.5, 2662.,  26.),
    ("Fornax",           -13.4,  710., 147.),
    ("Leo I",            -12.0,  251., 254.),
    ("Sculptor",         -11.1,  283.,  86.),
    ("Leo II",            -9.8,  176., 233.),
    ("Sextans",           -9.3,  695.,  86.),
    ("Carina",            -9.1,  250., 105.),
    ("Draco",             -8.8,  221.,  76.),
    ("Ursa Minor",        -8.8,  181.,  76.),
    ("Canes Venatici I",  -8.6,  564., 218.),
    ("Crater II",         -8.2, 1066., 117.),
    ("Hercules",          -6.6,  330., 132.),
    ("Bootes I",          -6.3,  242.,  66.),
    ("Leo IV",            -5.8,  206., 154.),
    ("Ursa Major I",      -5.5,  319.,  97.),
    ("Ursa Major II",     -4.2,  149.,  32.),
    ("Coma Berenices",    -4.1,   77.,  44.),
    ("Reticulum II",      -3.9,   51.,  32.),
    ("Willman 1",         -2.7,   25.,  38.),
    ("Segue 1",           -1.5,   29.,  23.),
]

# (M_V, rhalf/pc, d_helio/kpc) as a plain float array, for plotting.
MW_DWARFS = np.array([row[1:] for row in _MW_DWARFS], dtype=float)

MW_DWARF_NAMES = [row[0] for row in _MW_DWARFS]


def mw_dwarfs():
    """The table above as a DataFrame: ``name, mv, rhalf, d_helio``."""
    import pandas as pd
    return pd.DataFrame(_MW_DWARFS, columns=["name", "mv", "rhalf", "d_helio"])


# ---------------------------------------------------------------------------
# The published Milky Way satellite selection function
# ---------------------------------------------------------------------------
# Drlica-Wagner et al. 2020 (MW Satellite Census I, arXiv:1912.03302) and
# Tan et al. 2025 (DELVE MW Satellite Census I, arXiv:2509.12313) both
# characterize their searches by injecting large numbers of simulated satellites
# into the real survey data and recovering them with the real pipeline. Both
# then summarize the result with the *same* two-line analytic model: at fixed
# heliocentric distance, the locus of 50% detection probability in the
# (M_V, r_1/2) plane is
#
#     log10(r_1/2 / pc) = A0 / (M_V - M_V0) + log10(r_1/2,0),
#
# with three distance-dependent constants per survey. Equivalently, solving for
# the limiting magnitude of a satellite of a given size,
#
#     M_V,50 = M_V0 + A0 / (log10 r_1/2 - log10 r_1/2,0).
#
# Since log10(r_1/2,0) ~ 3.8-4.5, i.e. 6-30 kpc, the denominator is negative for
# any real dwarf and M_V,50 is always brighter than M_V0.
#
# Columns: heliocentric distance (kpc), A0, M_V0 (mag), log10(r_1/2,0 / pc).
MW_CENSUS_P50 = {
    # Drlica-Wagner+20 Table 5. DES Y3: ~5,000 deg^2, g ~ 24.3 (10 sigma).
    "des_y3": np.array([
        [11.3, 21.5,  7.8, 3.8],
        [22.6, 24.1,  8.3, 4.2],
        [45.2, 17.2,  5.2, 4.3],
        [90.5,  8.6,  1.2, 4.1],
        [181.,  6.6, -1.1, 4.1],
        [362.,  6.3, -2.3, 4.3],
    ]),
    # Drlica-Wagner+20 Table 5. PS1 DR1: g ~ 22.5 (10 sigma) -- the shallow one.
    "ps1_dr1": np.array([
        [11.3, 22.8,  7.1, 4.0],
        [22.6, 19.0,  5.0, 4.1],
        [45.2, 14.1,  1.8, 4.2],
        [90.5, 11.0, -0.3, 4.3],
        [181.,  7.5, -2.2, 4.2],
        [362.,  6.8, -4.0, 4.4],
    ]),
    # Tan+25, 50% detectability contour table. DES Y6: g ~ 24.7 (10 sigma) -- the deepest.
    "des_y6": np.array([
        [20.8,  20.3,  7.9, 4.1],
        [35.6,  23.9,  7.9, 4.5],
        [61.3,  17.7,  5.1, 4.5],
        [104.7, 13.0,  2.4, 4.5],
        [179.0, 10.8,  0.6, 4.5],
        [305.9,  9.4, -0.5, 4.5],
    ]),
    # Tan+25, same table. DELVE DR3: g ~ 24.2, but a deliberately stricter detection
    # threshold to suppress false positives, so its sensitivity lands between
    # DES Y3 and PS1 rather than next to DES Y6. Depth is not the whole story;
    # the purity requirement is part of the selection function.
    "delve_dr3": np.array([
        [20.8,  15.4,  5.6, 3.9],
        [35.6,  10.2,  3.1, 3.8],
        [61.3,  10.7,  2.1, 4.1],
        [104.7, 12.1,  0.8, 4.5],
        [179.0,  7.9, -1.3, 4.3],
        [305.9,  7.9, -2.0, 4.5],
    ]),
}

# Whole sky, square degrees.
SKY_AREA_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2

# The Tan+25 census footprint, already de-overlapped by the authors: DELVE DR3
# excludes the deeper DES Y6 regions, and PS1 covers what neither DECam survey
# reached. The three add to ~27,700 deg^2 = 67% of the celestial sphere and 91%
# of the sky at |b| >= 15 deg. The remaining third of the sky -- mostly the
# Galactic plane -- has no census at all.
MW_CENSUS_FOOTPRINT = (
    ("des_y6",    4800.0),
    ("delve_dr3", 12000.0),
    ("ps1_dr1",   10900.0),
)

# Distances over which Tan+25 consider a satellite detectable at all: below
# 16 kpc their algorithms are not optimized, above 400 kpc is beyond the Milky
# Way halo.
MW_CENSUS_DISTANCE_RANGE = (16.0, 400.0)

# Tan+25's headline result, for comparison: the completeness-corrected total
# satellite population in a precisely stated box.
TAN25_POPULATION = dict(n=265, lo=47, hi=79, n_recovered=49, n_known_in_footprint=62,
                        mv_range=(-20.0, 0.0), rhalf_range=(15.0, 3000.0),
                        dgc_range=(10.0, 300.0))


def census_p50_mv(rhalf, d_helio, survey="delve_dr3"):
    """
    ``M_V`` at which a satellite of half-light radius ``rhalf`` (pc) at distance
    ``d_helio`` (kpc) has a 50% chance of being detected.

    The three fit constants are interpolated linearly in ``log10 D`` between the
    six tabulated distances, which is what Drlica-Wagner+20 recommend ("the
    parameters ... vary smoothly as a function of distance, and interpolating
    between them can provide a reasonable approximation"). Outside the tabulated
    range the parameters are held fixed at the end values -- use
    ``MW_CENSUS_DISTANCE_RANGE`` to zero the probability instead of trusting an
    extrapolation.

    ``rhalf`` is the **azimuthally averaged** (circularized) half-light radius,
    which is what both papers inject and recover. Using a semi-major axis here
    would make every satellite look larger, and therefore harder to find, than
    the calibration assumed.
    """
    tab = MW_CENSUS_P50[survey]
    lgd = np.log10(np.clip(np.asarray(d_helio, dtype=float), tab[0, 0], tab[-1, 0]))
    lgd_tab = np.log10(tab[:, 0])
    a0 = np.interp(lgd, lgd_tab, tab[:, 1])
    mv0 = np.interp(lgd, lgd_tab, tab[:, 2])
    lgr0 = np.interp(lgd, lgd_tab, tab[:, 3])
    # The contour is only defined for r_1/2 < r_1/2,0 (6-30 kpc); the clip keeps
    # the branch on the right side for absurdly extended inputs.
    denom = np.minimum(np.log10(np.asarray(rhalf, dtype=float)) - lgr0, -0.05)
    return mv0 + a0 / denom


# Tan+25's inferred size-luminosity relation for the **total** (completeness-
# corrected) Milky Way satellite population, from their empirical model:
#
#     <log10(r_1/2 / pc)> = z_SL + m_SL * (M_V + 6),   scatter sigma_SL dex.
#
# This is the curve a mock's *intrinsic* sizes should be compared against.
# Comparing them with the observed dwarfs instead is a trap at the faint end:
# compact satellites are easier to find, so the detected population is
# systematically more compact than the real one. Tan+25 say so explicitly, and
# their Figure 11 shows the two curves separating below M_V ~ -4.
TAN25_SIZE_LUMINOSITY = dict(z=2.07, z_err=0.04, m=-0.12, m_err=0.01,
                             sigma=0.24, sigma_err=0.03)


def tan25_mean_log_rhalf(mv):
    """Mean ``log10(r_1/2 / pc)`` of the *total* MW satellite population at ``M_V``."""
    p = TAN25_SIZE_LUMINOSITY
    return p["z"] + p["m"] * (np.asarray(mv, dtype=float) + 6.0)


# ---------------------------------------------------------------------------
# The census box, the measured softening, and the empirical luminosity function
# ---------------------------------------------------------------------------

# Tan+25 state their result inside a precisely defined box. Every comparison in
# the tutorial's Milky Way branch is cut to it -- the mock, the injection-
# recovery simulations, and the observed sample alike. Note ``d_gc`` bounds the
# *population* while ``MW_CENSUS_DISTANCE_RANGE`` bounds *detectability*: they
# are different cuts and both apply.
TAN25_BOX = dict(mv=(-20.0, 0.0), rhalf=(15.0, 3000.0), d_gc=(10.0, 300.0))


# The width, in magnitudes, over which the 50% contour softens from "not
# detected" to "detected".
#
# Measured, by fitting
#
#     P_det = logistic((M_V,50(r_1/2, D) - M_V) / w)
#
# to the released injection-recovery simulations -- the same simulations the
# published contour was fit to. Two things came out of that fit and both are
# worth knowing:
#
#   * the logistic *shape* is right. With a single global width per survey the
#     model reproduces the simulations' own recovered counts to 0.3%, and to
#     better than 4% in every octave of distance.
#   * the published contour really is the 50% contour. The fitted offset is
#     -0.03 mag (DES Y6) and -0.01 mag (DELVE DR3), and the measured detection
#     fraction within 0.15 mag of the contour is 0.49 and 0.51.
#
# The fit is over the simulations as shipped, i.e. already cut to TAN25_BOX;
# notebook 05 reproduces these numbers from ``io.load_census_sims()``. Fitting
# the uncut release instead gives 0.36 / 0.41 -- the very compact injections
# below r_1/2 = 15 pc soften the transition slightly.
#
# PS1 has no simulations in this release -- theirs belong to Drlica-Wagner+20 --
# so it borrows the DELVE DR3 width, which is the closer of the two in depth.
CENSUS_WIDTH = {"des_y6": 0.32, "delve_dr3": 0.38, "ps1_dr1": 0.38,
                "des_y3": 0.38}

# The width drifts with distance, from ~0.42-0.50 mag at 16-32 kpc to ~0.25-0.29
# mag beyond 128 kpc: distant satellites are resolved into fewer member stars, so
# the transition from findable to not is sharper. Using one number per survey
# instead costs under 4% in any distance octave, which is why we do.


# Brighter than this, treat the census as complete over the **whole sky**, not
# just inside its footprint.
#
# The footprint average is the right thing for a faint satellite: nobody would
# have found it outside the searched area, because nobody looked there deeply
# enough. It is the wrong thing for a classical dwarf. Fornax, Sculptor, Draco
# and their kind were found on photographic plates covering the whole sky,
# decades before DES existed, and the census counts them; a model that gives
# them P_det = 0.67 because two thirds of the sky was searched will
# under-predict the bright end by exactly that factor.
#
# The threshold is a judgement call and the tutorial treats it as one:
#
#   * Tan+25's own released code makes the weaker assumption, P_det = 1 only for
#     M_V < -12.5 -- the LMC, the SMC and Sagittarius.
#   * -8 is the stronger, and more common, claim that nothing brighter than a
#     classical dwarf is still hiding. It is *nearly* true, and the exceptions
#     are instructive: Crater II (M_V = -8.2) was found in 2016 and Antlia II
#     (M_V = -9.0) in 2019, both because they are enormous and diffuse rather
#     than because they are faint. Neither is in the Galactic plane.
#
# Notebook 05 runs the comparison both ways; the choice moves the bright end by
# ~1.5x and nothing else.
CENSUS_BRIGHT_ALL_SKY_MV = -8.0


def tan25_luminosity_function(mv, chain, cumulative=True):
    """
    The **completeness-corrected** Milky Way satellite luminosity function of
    Tan+25, evaluated for every sample in their posterior ``chain``.

    ``chain`` is the array loaded by ``io.load_census_chain``, columns
    ``[beta, rcore, sigma_sl, zp, sp, n_total]``.

    Their empirical model is a power law in luminosity,

        dN/dM_V ~ 10^(-beta M_V),

    normalized so that ``n_total`` satellites lie in ``TAN25_BOX['mv']``. The
    cumulative count brighter than ``M_V`` is therefore

        N(<M_V) = n_total * (10^(beta M_V) - 10^(beta M_V,min))
                          / (10^(beta M_V,max) - 10^(beta M_V,min)).

    Returns an array of shape ``(len(chain), len(mv))``: the posterior predictive
    for the luminosity function, from which ``tan25_lf_quantiles`` takes bands.

    **This is the total population, not the detected one.** It is the right
    thing to compare a mock's *intrinsic* satellites against. To compare against
    a mock's *detected* satellites, use the observed census sample instead.
    """
    mv = np.atleast_1d(np.asarray(mv, dtype=float))
    chain = np.atleast_2d(np.asarray(chain, dtype=float))
    beta = chain[:, 0][:, None]
    n_total = chain[:, 5][:, None]
    lo, hi = TAN25_BOX["mv"]

    num = 10.0 ** (beta * mv[None, :]) - 10.0 ** (beta * lo)
    den = 10.0 ** (beta * hi) - 10.0 ** (beta * lo)
    n_cum = n_total * np.clip(num / den, 0.0, 1.0)
    if cumulative:
        return n_cum
    return np.diff(n_cum, axis=1)


def tan25_lf_quantiles(mv, chain, q=(16, 50, 84)):
    """Percentiles of ``tan25_luminosity_function`` across the posterior."""
    return np.percentile(tan25_luminosity_function(mv, chain), q, axis=0)


def tan25_radial_profile(d_gc, chain, cumulative=True):
    """
    The **completeness-corrected** galactocentric radial distribution of Tan+25.

    Their empirical model is a cored profile in number density,

        n(D) ~ D^-0 (D + r_core)^-3    ->    dN/dD ~ D^2 (D + r_core)^-3,

    with ``r_core`` (column 1 of the chain, in kpc) free and the population
    normalized to ``n_total`` inside ``TAN25_BOX['d_gc']``. The cumulative count
    inside ``D`` follows from

        F(D) = ln(D + r_c) + 2 r_c / (D + r_c) - r_c^2 / [2 (D + r_c)^2],

    which is the integral of ``D^2 (D + r_c)^-3`` done by substituting
    ``u = D + r_c``. Then ``N(<D) = n_total [F(D) - F(D_min)] / [F(D_max) - F(D_min)]``.

    Returns ``(len(chain), len(d_gc))``, matching ``tan25_luminosity_function``.
    Like it, this is the *total* population -- compare it to a mock's intrinsic
    satellites, and compare the census sample to the mock's detected ones.
    """
    d = np.atleast_1d(np.asarray(d_gc, dtype=float))
    chain = np.atleast_2d(np.asarray(chain, dtype=float))
    rc = chain[:, 1][:, None]
    n_total = chain[:, 5][:, None]
    lo, hi = TAN25_BOX["d_gc"]

    def _f(x):
        u = x + rc
        return np.log(u) + 2.0 * rc / u - 0.5 * rc ** 2 / u ** 2

    frac = (_f(np.clip(d, lo, hi)[None, :]) - _f(lo)) / (_f(hi) - _f(lo))
    n_cum = n_total * np.clip(frac, 0.0, 1.0)
    if cumulative:
        return n_cum
    return np.diff(n_cum, axis=1)


def tan25_radial_quantiles(d_gc, chain, q=(16, 50, 84)):
    """Percentiles of ``tan25_radial_profile`` across the posterior."""
    return np.percentile(tan25_radial_profile(d_gc, chain), q, axis=0)
