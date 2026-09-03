"""
Layer 3c: the survey selection function, and the one high-level API.

Everything upstream of this module produces galaxies that *exist*. This module
decides which of them are *seen*. In the plan's notation,

    N_pred = INT tau(D) P(D | theta) dD,

where D are observable galaxy properties, theta the model parameters, and
tau(D) the survey selection function. A Monte Carlo mock evaluates that integral
by sampling galaxies and detecting them probabilistically -- which is all
``apply_completeness`` does.

Selection is not one number. It factorizes, at least, as

    P_obs = P_footprint x P_detect x P_membership,

and the first factor is pure geometry: a survey that only imaged the inner half
of a halo cannot find satellites in the outer half, however bright they are.
Following the plan we implement the first two and leave membership to the
discussion.

Two branches, matching the two datasets:

    external hosts (ELVES)  a one-dimensional completeness in Mstar or M_V,
                            inside an aperture the survey actually covered
    Milky Way (LVDB)        P_det(M_V, r_half, D_helio), because a resolved-star
                            search needs both enough member stars *and* enough
                            contrast against the foreground
"""

import numpy as np
from scipy.special import erf

from .galaxy_halo import populate_subhalos
from .observables import M_V_SUN, add_observables, effective_surface_brightness
from .projection import place_mw_observer, project_satellites
from .reference import (CENSUS_BRIGHT_ALL_SKY_MV, CENSUS_WIDTH,
                        MW_CENSUS_DISTANCE_RANGE, MW_CENSUS_FOOTPRINT,
                        SKY_AREA_DEG2, census_p50_mv)

__all__ = [
    "apply_aperture",
    "detection_probability_mstar",
    "detection_probability_mv",
    "n_resolved_stars",
    "mw_detection_probability",
    "census_sky_weights",
    "census_detection_probability",
    "apply_completeness",
    "apply_survey_selection",
    "mock_observe",
    "ELVES_SURVEY",
    "MW_SURVEY",
    "MW_SURVEY_ISOTROPIC",
    "MW_SURVEY_TOY",
]


# ---------------------------------------------------------------------------
# P_footprint: the aperture
# ---------------------------------------------------------------------------
def apply_aperture(sats, f_rvir=None, r_max_kpc=None, radius_col="r_proj_los",
                   host_col="host_id", host_rvir_col="host_rvir"):
    """
    Keep satellites inside the survey's aperture.

    Exactly one of:

    ``f_rvir``     a multiple of each host's own virial radius -- the natural
                   theory aperture;
    ``r_max_kpc``  a radius in kpc, either one number for every host or a
                   mapping ``{host_id: radius}`` -- the natural *observational*
                   aperture, because a survey covers a fixed area on the sky,
                   not a fixed fraction of a halo.

    The distinction matters more than it looks. ELVES searched a median of
    1.15 host virial radii, but the spread runs from 0.4 to 1.9: a handful of
    hosts were barely covered inside half their halo. Applying "1 Rvir" to the
    mock while the data had 0.4 Rvir for some hosts is a selection mismatch that
    no amount of care with the galaxy-halo model will repair.
    """
    if (f_rvir is None) == (r_max_kpc is None):
        raise ValueError("pass exactly one of f_rvir, r_max_kpc")

    r = sats[radius_col].to_numpy(dtype=float)
    if f_rvir is not None:
        limit = f_rvir * sats[host_rvir_col].to_numpy(dtype=float)
    elif np.isscalar(r_max_kpc):
        limit = float(r_max_kpc)
    else:
        limit = sats[host_col].map(r_max_kpc).to_numpy(dtype=float)

    return sats[r < limit].copy()


# ---------------------------------------------------------------------------
# Selection level 1: a one-dimensional completeness (external hosts)
# ---------------------------------------------------------------------------
def detection_probability_mstar(mstar, logMstar50=5.6, sigma_det=0.25, p_max=1.0):
    """
    Probability of detecting a satellite of stellar mass ``mstar``.

    The same smooth step used for occupation, now describing the *survey*
    rather than galaxy formation:

        P_det = p_max/2 * [1 + erf((log Mstar - log Mstar50) / (sqrt(2) sigma_det))].

    ``p_max`` < 1 is the plateau far above the limit: even a bright satellite is
    missed sometimes, and no survey is 100% complete anywhere.

    Note the shape is identical to ``galaxy_halo.occupation_fraction``. That is
    not a coincidence so much as a warning -- a deficit of faint satellites can
    be produced by moving either curve, and a single count cannot say which.
    """
    lgm = np.log10(np.asarray(mstar, dtype=float))
    if sigma_det <= 0:
        return np.where(lgm > logMstar50, p_max, 0.0)
    return 0.5 * p_max * (1.0 + erf((lgm - logMstar50) / (np.sqrt(2.0) * sigma_det)))


def detection_probability_mv(mv, mv50=-9.0, sigma_det=0.3, p_max=1.0):
    """
    The same curve in absolute magnitude, where surveys usually quote their limits.

    Magnitudes run backwards, so the sign flips: bright means ``mv`` **below**
    ``mv50``. Converting a limit in M_V into a limit in Mstar requires a
    mass-to-light ratio, so the two versions of "the same" completeness cut do
    not select the same galaxies unless Upsilon_V matches the survey's.
    """
    mv = np.asarray(mv, dtype=float)
    if sigma_det <= 0:
        return np.where(mv < mv50, p_max, 0.0)
    return 0.5 * p_max * (1.0 + erf((mv50 - mv) / (np.sqrt(2.0) * sigma_det)))


# ---------------------------------------------------------------------------
# Selection level 2: the Milky Way, P_det(M_V, r_half, D)
# ---------------------------------------------------------------------------
# Number of member stars brighter than absolute magnitude M, per solar
# luminosity of an old, metal-poor population:
#
#     n(<M) / (L_V/Lsun) = 10^(LF_SLOPE * (M - LF_ZERO)),  capped at 1.
#
# Calibrated on two real objects rather than on an isochrone, so the numbers can
# be checked by hand:
#
#   Bootes I   M_V = -6.0 (L_V = 2.2e4 Lsun) at 66 kpc has ~40 members brighter
#              than M_V ~ +0.4          -> n/L = 1.8e-3
#   Segue 1    M_V = -1.3 (L_V = 2.8e2 Lsun) at 23 kpc has ~70 members brighter
#              than M_V ~ +4.9          -> n/L = 0.25
#
# Those two points give a slope of 0.45 dex per magnitude and a zero point at
# M = 6.5. The cap at n = L/Lsun is where the approximation runs out: it cannot
# be right that a population contains more stars than it has solar luminosities.
LF_SLOPE = 0.45
LF_ZERO = 6.5


def n_resolved_stars(mv, d_helio, m_lim=22.5, lf_slope=LF_SLOPE, lf_zero=LF_ZERO):
    """
    Member stars brighter than apparent magnitude ``m_lim``, at distance ``d_helio`` (kpc).

    This is the variable that actually carries the distance dependence of a
    resolved-star search. Surface brightness does not: it is distance
    independent by construction. What fades with distance is the *number of
    stars a survey can measure*, since the distance modulus pushes the
    luminosity function past the survey limit.

    ``m_lim = 22.5`` stands in for a DES/PS1/DELVE-like depth for reliable
    stellar photometry. It is one number standing in for a patchwork of surveys
    of different depths over different parts of the sky.
    """
    mv = np.asarray(mv, dtype=float)
    d = np.asarray(d_helio, dtype=float)
    lum = 10.0 ** (-0.4 * (mv - M_V_SUN))
    dm = 5.0 * np.log10(np.maximum(d, 1e-6) * 1e3 / 10.0)
    frac = np.minimum(10.0 ** (lf_slope * (m_lim - dm - lf_zero)), 1.0)
    return lum * frac


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def mw_detection_probability(mv, rhalf, d_helio, m_lim=22.5, n_min=20.0,
                             w_n=0.2, mu_lim=30.5, w_mu=0.7, f_sky=0.75,
                             lf_slope=LF_SLOPE, lf_zero=LF_ZERO):
    """
    A transparent toy for the Milky Way satellite selection function.

    Two conditions, multiplied, because a resolved overdensity has to be both
    *populated* and *distinguishable*:

        P_det = f_sky
              x logistic( (log10 N_star - log10 n_min) / w_n )       enough stars
              x logistic( (mu_lim - <mu_V>_e) / w_mu )               enough contrast

    ``rhalf`` in pc, ``d_helio`` in kpc.

    This is **not** the DES/PS1/DELVE selection function, and it is not a
    substitute for injection-recovery. The two thresholds were placed by eye
    against the faint envelope of the known Milky Way dwarfs in the Local Volume
    Database -- which is circular, since that envelope is itself shaped by the
    selection we are trying to model. It gives the right *structure* and roughly
    the right location; treat the coefficients as illustrative.

    What the two terms buy, separately:

    - the star-count term makes the limiting luminosity a strong function of
      distance (M_V ~ 0 at 25 kpc, ~ -4 at 150 kpc, ~ -5.6 at 300 kpc), which is
      what the real census shows;
    - the surface-brightness term makes it depend on size at fixed luminosity,
      which is why Crater II sat undiscovered until 2016.

    ``f_sky`` is the fraction of the sky searched to this depth -- P_footprint
    for the Milky Way branch. It is a flat prefactor here; in reality it is a
    depth that varies across the sky, and the Galactic plane is not searched at
    all.
    """
    n_star = n_resolved_stars(mv, d_helio, m_lim=m_lim, lf_slope=lf_slope,
                              lf_zero=lf_zero)
    mu = effective_surface_brightness(mv, rhalf)

    with np.errstate(divide="ignore"):
        p_stars = _logistic((np.log10(np.maximum(n_star, 1e-30))
                             - np.log10(n_min)) / w_n)
    p_contrast = _logistic((mu_lim - mu) / w_mu)
    return f_sky * p_stars * p_contrast


# ---------------------------------------------------------------------------
# The published selection function
# ---------------------------------------------------------------------------
_FOOTPRINT_MAP = None


def census_sky_weights(ra, dec, footprint_map=None):
    """
    Which census survey covers each sky position, and how much of it.

    Returns ``{survey: fraction}`` for the three census surveys, where the
    fraction is how much of that position's 0.64 deg^2 cell the survey usably
    searched. The three sum to at most one -- Tan+25 de-overlap their footprints,
    so a given piece of sky belongs to exactly one survey -- and to zero over the
    third of the sky nobody searched.

    ``ra``, ``dec`` in degrees. ``footprint_map`` defaults to
    ``io.load_census_footprint()``, cached after the first call.

    This is the position-resolved version of the area weighting in
    ``census_detection_probability``. Averaged over an isotropic population the
    two give the same answer; a single realization sees the difference, because
    a footprint with holes in it produces scatter that an area fraction cannot.
    """
    global _FOOTPRINT_MAP
    if footprint_map is None:
        if _FOOTPRINT_MAP is None:
            from .io import load_census_footprint
            _FOOTPRINT_MAP = load_census_footprint()
        footprint_map = _FOOTPRINT_MAP

    cov = footprint_map["coverage"]
    n_sindec, n_ra = cov.shape[1:]
    i = np.clip((np.asarray(ra, dtype=float) % 360.0) / 360.0 * n_ra,
                0, n_ra - 1e-9).astype(int)
    sind = np.sin(np.radians(np.asarray(dec, dtype=float)))
    j = np.clip((sind + 1.0) / 2.0 * n_sindec, 0, n_sindec - 1e-9).astype(int)
    return {name: cov[k][j, i] for k, name in enumerate(footprint_map["surveys"])}


def census_detection_probability(mv, rhalf, d_helio, width=None, survey=None,
                                 ra=None, dec=None,
                                 footprint=MW_CENSUS_FOOTPRINT,
                                 d_range=MW_CENSUS_DISTANCE_RANGE,
                                 bright_all_sky_mv=None):
    """
    Detection probability from the **published** Milky Way census selection
    functions (Drlica-Wagner et al. 2020; Tan et al. 2025).

    ``rhalf`` in pc (circularized), ``d_helio`` in kpc.

    Both papers derive their selection function by injecting simulated
    satellites into the real survey data and recovering them with the real
    pipeline, then summarize the result as the 50% detection contour
    ``reference.census_p50_mv``. This function turns that contour into a
    probability by softening it into a logistic of width ``width`` magnitudes:

        P_det = logistic((M_V,50(r_1/2, D) - M_V) / width).

    **The softening is measured, not assumed.** ``width=None``, the default,
    takes the per-survey value in ``reference.CENSUS_WIDTH``, fitted to the
    released injection-recovery simulations (``io.load_census_sims``) -- the same
    simulations the contour itself was fit to. With one width per survey the
    model reproduces those simulations' own recovered counts to 1%. Pass a number
    to override it, which is worth doing once to see how little it matters:
    0.25 to 1.0 mag moves the predicted count by under 10%, because the
    population is steep and most satellites sit far from the contour on one side
    or the other.

    The trained XGBoost classifiers remain the more accurate tool for a real
    analysis (https://github.com/des-science/mw-sats and
    https://github.com/delve-survey/delve_mw_census), because they also carry
    the dependence on foreground stellar density within a field.

    Three ways of handling the sky, in increasing order of realism:

    ``survey="des_y6"``   one field's sensitivity everywhere. Use it to compare
                          surveys, not to predict a count.
    default               averaged over the census footprint, weighted by area,
                          ``P_det = sum_s (Omega_s / 4pi) P_det,s``. Correct for
                          the *mean* of an isotropic mock population, and it is
                          what a mock with no sky positions has to use.
    ``ra=``, ``dec=``     look the surveys up at each object's own position in
                          the real mask (``census_sky_weights``). Same mean,
                          correct scatter. ``projection.place_mw_observer``
                          provides the columns.

    None of the three is right for the *known* satellites, which are
    concentrated where the deep data are because that is where they were found.

    ``bright_all_sky_mv`` sets a magnitude brighter than which ``P_det = 1``
    regardless of the footprint, on the grounds that the classical dwarfs were
    catalogued all-sky long before DES. ``None`` disables it; see
    ``reference.CENSUS_BRIGHT_ALL_SKY_MV`` for why the tutorial uses -8 and why
    that is a judgement call rather than a measurement.

    Outside ``d_range`` the probability is zero, following Tan+25: below 16 kpc
    their algorithms are not optimized, and 400 kpc is beyond the halo.
    """
    mv = np.asarray(mv, dtype=float)
    d = np.asarray(d_helio, dtype=float)
    shape = np.broadcast(mv, np.asarray(rhalf, dtype=float), d).shape

    if survey is not None:
        weights = {survey: np.ones(shape)}
    elif ra is not None and dec is not None:
        weights = census_sky_weights(ra, dec)
    else:
        weights = {name: area / SKY_AREA_DEG2 for name, area in footprint}

    p = np.zeros(shape)
    for name, w in weights.items():
        wd = CENSUS_WIDTH[name] if width is None else width
        mv50 = census_p50_mv(rhalf, d, survey=name)
        p = p + w * _logistic((mv50 - mv) / wd)

    if bright_all_sky_mv is not None:
        p = np.where(mv < bright_all_sky_mv, 1.0, p)

    lo, hi = d_range
    return np.where((d >= lo) & (d <= hi), p, 0.0)


# ---------------------------------------------------------------------------
# Turning a probability into a catalog
# ---------------------------------------------------------------------------
def apply_completeness(sats, probability, rng=None, prob_col="p_det",
                       flag_col="detected"):
    """
    Draw a Bernoulli detection per satellite. Returns a copy with two columns added.

    This is the Monte Carlo evaluation of ``INT tau(D) P(D|theta) dD``: sample
    the galaxies, then keep each with probability ``tau``. Doing it as a *draw*
    rather than a weight is deliberate -- it keeps the mock a catalog, with the
    right integer counting noise, so a mock host with a predicted 4.3 satellites
    sometimes has 2 and sometimes 7, exactly as a real one does.
    """
    rng = np.random.default_rng(rng)
    out = sats.copy()
    p = np.asarray(probability, dtype=float)
    out[prob_col] = p
    out[flag_col] = rng.uniform(size=len(out)) < p
    return out


# ---------------------------------------------------------------------------
# Survey definitions
# ---------------------------------------------------------------------------
# ELVES (Carlsten et al. 2022). The satellite catalog is cut at M_V = -9, and
# the survey is quoted as roughly 90% complete above it inside the covered area,
# so the plateau is set below one. sigma_det is a stand-in for the fact that the
# limit is not a step: near it, detection depends on surface brightness too.
ELVES_SURVEY = dict(
    geometry="projected",
    aperture=dict(f_rvir=1.0),
    selection=dict(kind="mv", mv50=-9.0, sigma_det=0.3, p_max=0.9),
    requires_mv=True,
)

# Milky Way. No aperture in the survey sense -- the whole halo is in front of
# us -- but only part of the sky is searched, and to three different depths.
# MW_SURVEY uses the *published* selection function; MW_SURVEY_TOY is the
# from-scratch version of ``mw_detection_probability``, kept so the two can be
# put side by side.
MW_SURVEY = dict(
    geometry="mw",
    aperture=dict(r_max_kpc=300.0),
    selection=dict(kind="census", width=None, use_sky=True,
                   bright_all_sky_mv=CENSUS_BRIGHT_ALL_SKY_MV),
    requires_mv=True,
)

# The same thing with the sky switched off: the footprint enters only as an area
# fraction, and nothing is complete outside it. Useful for showing what the mask
# and the bright-end assumption each buy.
MW_SURVEY_ISOTROPIC = dict(
    geometry="mw",
    aperture=dict(r_max_kpc=400.0),
    selection=dict(kind="census", width=None, use_sky=False,
                   bright_all_sky_mv=None),
    requires_mv=True,
)

MW_SURVEY_TOY = dict(
    geometry="mw",
    aperture=dict(r_max_kpc=300.0),
    selection=dict(kind="mw", m_lim=22.5, n_min=20.0, w_n=0.2,
                   mu_lim=30.5, w_mu=0.7, f_sky=0.75),
    requires_mv=True,
)


def apply_survey_selection(sats, selection, rng=None):
    """
    Evaluate the appropriate detection probability and draw from it.

    ``selection`` is the ``"selection"`` sub-dictionary of a survey definition;
    its ``"kind"`` picks the branch. ``kind="none"`` marks everything detected,
    which is how the intrinsic population is carried through the same code path
    as the observed one.
    """
    params = dict(selection)
    kind = params.pop("kind", "none")

    if kind == "none":
        p = np.ones(len(sats))
    elif kind == "mstar":
        p = detection_probability_mstar(sats["mstar"].to_numpy(), **params)
    elif kind == "mv":
        p = detection_probability_mv(sats["mv"].to_numpy(), **params)
    elif kind == "census":
        # `use_sky` asks for the real mask, which needs sky positions --
        # `place_mw_observer` puts them there. Without them, fall back to the
        # area-weighted footprint average rather than failing.
        use_sky = params.pop("use_sky", False)
        if use_sky and {"ra", "dec"}.issubset(sats.columns):
            params["ra"] = sats["ra"].to_numpy()
            params["dec"] = sats["dec"].to_numpy()
        p = census_detection_probability(sats["mv"].to_numpy(),
                                         sats["rhalf"].to_numpy(),
                                         sats["d_helio"].to_numpy(), **params)
    elif kind == "mw":
        p = mw_detection_probability(sats["mv"].to_numpy(),
                                     sats["rhalf"].to_numpy(),
                                     sats["d_helio"].to_numpy(), **params)
    else:
        raise ValueError(f"unknown selection kind {kind!r}")

    return apply_completeness(sats, p, rng=rng)


# ---------------------------------------------------------------------------
# The whole chain, in one call
# ---------------------------------------------------------------------------
def mock_observe(halo_catalog, occupation_params=None, shmr_params=None,
                 observable_params=None, survey_params=None, rng=None,
                 detected_only=True, mpeak_min=1e8):
    """
    Subhalo catalog -> mock observed satellite catalog.

    ::

        halo catalog
          -> populate_subhalos     occupation + SHMR
          -> add_observables       M_V, r_half, mu
          -> geometry              projection, or a Solar-position observer
          -> apply_aperture        P_footprint
          -> apply_survey_selection  P_detect
          -> mock observed catalog

    ``survey_params`` is a complete survey definition -- ``ELVES_SURVEY`` or
    ``MW_SURVEY``, or a copy of one with entries replaced, e.g.
    ``dict(ELVES_SURVEY, aperture=dict(r_max_kpc=r_by_host))``. It is used as
    given rather than merged with a default, so a survey is always described in
    one visible place. ``observable_params`` may carry ``"mass_to_light"`` and
    ``"size"``.

    With ``detected_only=False`` the full post-aperture table comes back instead,
    carrying ``p_det`` and ``detected``. That is the version to use for any
    intrinsic-versus-observed comparison: both populations then come from the
    *same* random draw, so the difference between the curves is the selection
    function and nothing else.

    ``mpeak_min`` is a floor on peak halo mass, and it is not cosmetic. The
    catalog reaches 10^7, but both published SHMRs are calibrated only above
    10^8; extrapolating a slope-2 power law the rest of the way predicts
    galaxies of a few solar masses, which is not a prediction so much as an
    artifact. Occupation hides most of them and the survey selection hides the
    rest, so they never reach a plot -- but they should not be in the table
    either. Pass ``None`` to keep the whole catalog.

    One ``rng`` threads through every stochastic step, so the whole mock is
    reproducible from a single seed.
    """
    rng = np.random.default_rng(rng)
    if mpeak_min is not None:
        halo_catalog = halo_catalog[halo_catalog["mpeak"] > mpeak_min]
    survey = ELVES_SURVEY if survey_params is None else survey_params
    obs = dict(observable_params or {})

    sats = populate_subhalos(halo_catalog, occupation_params=occupation_params,
                             shmr_params=shmr_params, rng=rng)
    sats = add_observables(sats,
                           mass_to_light_params=obs.get("mass_to_light"),
                           size_params=obs.get("size"), rng=rng)

    if survey["geometry"] == "projected":
        sats = project_satellites(sats, rng=rng)
        radius_col = "r_proj_los"
    elif survey["geometry"] == "mw":
        sats = place_mw_observer(sats, rng=rng)
        radius_col = "r_3d"
    else:
        raise ValueError(f"unknown geometry {survey['geometry']!r}")

    sats = apply_aperture(sats, radius_col=radius_col, **survey["aperture"])
    sats = apply_survey_selection(sats, survey["selection"], rng=rng)

    return sats[sats["detected"]].copy() if detected_only else sats
