"""
Layer 3a of the forward model: from stellar mass to *observable* galaxy properties.

The galaxy-halo connection in ``galaxy_halo.py`` ends at ``Mstar``. Everything a
survey actually measures -- a magnitude, an angular size, a surface brightness --
sits downstream of that, and depends on stellar-population and structural
assumptions that have nothing to do with the SHMR. Keeping them in a separate
module is the point of Sections 4 and 5.

Units, fixed once and used everywhere:

    Mstar      Msun
    M_V        absolute V magnitude (Vega)
    rhalf      3D-projected half-light radius, in **pc**
    rvir_acc   halo virial radius at accretion, in **kpc** (as in the catalog)
    mu         mean V surface brightness within rhalf, mag arcsec^-2
"""

import numpy as np

__all__ = [
    "M_V_SUN",
    "SB_CONST",
    "mass_to_light_ratio",
    "stellar_mass_to_mv",
    "mv_to_stellar_mass",
    "assign_size",
    "effective_surface_brightness",
    "add_observables",
    "FIDUCIAL_MASS_TO_LIGHT",
    "FIDUCIAL_SIZE",
]


# Absolute V magnitude of the Sun (Willmer 2018, Vega system).
M_V_SUN = 4.83


# ---------------------------------------------------------------------------
# Fiducial parameters
# ---------------------------------------------------------------------------
# Upsilon_V = Mstar / L_V. An old, metal-poor, roughly single-burst dwarf
# population sits near 1.5-2; younger or more metal-rich ones go below 1. The
# whole plausible range 1.0-2.5 is only 0.4 dex = 1.0 mag, so this is a real but
# bounded systematic -- unlike the occupation threshold, which is a factor of 20.
FIDUCIAL_MASS_TO_LIGHT = dict(mass_to_light_v=1.5, sigma_log_ml=0.0)

# rhalf = 0.015 * Rvir is Kravtsov (2013); we write it as 1500 pc at a pivot of
# Rvir = 100 kpc with unit slope, which is the same relation in the plan's
# parameterization. Checked against the classic MW dwarfs in
# ``reference.MW_DWARFS``: the mock median tracks them to better than a factor
# of ~1.5 from M_V = -13 to -4. See notebook 03, Section 5.
FIDUCIAL_SIZE = dict(size_norm=1500.0, size_slope=1.0,
                     sigma_log_size=0.2, rvir_pivot=100.0)


# ---------------------------------------------------------------------------
# Section 4: Mstar -> M_V
# ---------------------------------------------------------------------------
def mass_to_light_ratio(mstar, mass_to_light_v=1.5, mass_to_light_bright=None,
                        mv_split=-8.0, split_width=0.0):
    """
    ``Upsilon_V`` per galaxy, optionally different for bright and faint dwarfs.

    With ``mass_to_light_bright=None`` (the default) this is just the constant
    ``mass_to_light_v``. Give it a value and Upsilon becomes a function of
    stellar mass, ``mass_to_light_bright`` above the split and
    ``mass_to_light_v`` below it, on the grounds that the classical dwarfs have
    younger and more metal-rich populations than the ultra-faints.

    The split is stated as a magnitude, ``mv_split``, but has to be applied in
    stellar mass -- Upsilon is what turns one into the other, so a threshold on
    the *output* magnitude would be circular. The boundary is therefore the
    stellar mass that ``mv_split`` corresponds to at the **faint** Upsilon.

    ``split_width`` is the width of the transition in dex of stellar mass, and
    **it should not be left at zero.** A hard step multiplies the luminosity of
    every galaxy above the split by ``mass_to_light_v / mass_to_light_bright``
    all at once, which tears a gap of

        2.5 log10(Upsilon_faint / Upsilon_bright)

    magnitudes in the luminosity function -- 0.44 mag for 1.5 and 1.0, and
    entirely an artifact of the parameterization. Any width above ~0.05 dex
    keeps ``M_V(Mstar)`` monotonic and closes the gap; 0.2-0.3 dex is a
    reasonable stand-in for how gradually stellar populations actually change.
    """
    mstar = np.asarray(mstar, dtype=float)
    if mass_to_light_bright is None:
        return float(mass_to_light_v)

    lg_split = np.log10(mass_to_light_v) - 0.4 * (mv_split - M_V_SUN)
    dlg = np.log10(np.maximum(mstar, 1e-30)) - lg_split
    if split_width <= 0:
        f = (dlg > 0).astype(float)
    else:
        f = 1.0 / (1.0 + np.exp(-np.clip(dlg / split_width, -500, 500)))
    return mass_to_light_v + (float(mass_to_light_bright) - mass_to_light_v) * f


def stellar_mass_to_mv(mstar, mass_to_light_v=1.5, sigma_log_ml=0.0, rng=None,
                       mass_to_light_bright=None, mv_split=-8.0, split_width=0.0):
    """
    Absolute V magnitude from stellar mass, through a mass-to-light ratio.

        L_V  = Mstar / Upsilon_V
        M_V  = M_V_sun - 2.5 log10(L_V / Lsun)

    With ``sigma_log_ml > 0``, ``log10 Upsilon_V`` is drawn per galaxy from a
    Gaussian of that width -- a stand-in for star-formation-history and
    metallicity variation between dwarfs.

    ``mass_to_light_bright``, ``mv_split`` and ``split_width`` make Upsilon
    luminosity dependent; see ``mass_to_light_ratio``. Note which way that moves
    a galaxy: **lowering** Upsilon at the bright end makes bright galaxies
    *brighter* by 2.5 log10(Upsilon_faint/Upsilon_bright), so it *raises*
    N(< M_V) there.

    Note this step is nearly a **relabelling of the x axis**: at fixed
    ``Upsilon_V`` it is one-to-one, so it moves no galaxy past another. It
    matters because the MW satellite literature is written in ``M_V``, and
    because surface brightness needs a luminosity.
    """
    mstar = np.asarray(mstar, dtype=float)
    ml = mass_to_light_ratio(mstar, mass_to_light_v=mass_to_light_v,
                             mass_to_light_bright=mass_to_light_bright,
                             mv_split=mv_split, split_width=split_width)
    if sigma_log_ml > 0:
        rng = np.random.default_rng(rng)
        ml = ml * 10.0 ** rng.normal(0.0, sigma_log_ml, size=np.shape(mstar))
    return M_V_SUN - 2.5 * np.log10(mstar / ml)


def mv_to_stellar_mass(mv, mass_to_light_v=1.5):
    """Inverse of ``stellar_mass_to_mv`` at fixed ``Upsilon_V``. Returns Msun."""
    mv = np.asarray(mv, dtype=float)
    return mass_to_light_v * 10.0 ** (-0.4 * (mv - M_V_SUN))


# ---------------------------------------------------------------------------
# Section 5: galaxy size
# ---------------------------------------------------------------------------
def assign_size(rvir_acc, size_norm=1500.0, size_slope=1.0, sigma_log_size=0.2,
                rvir_pivot=100.0, rng=None):
    """
    Half-light radius in **pc** from the halo's virial radius at accretion (kpc):

        rhalf = size_norm * (Rvir_acc / rvir_pivot)^size_slope * 10^delta,
        delta ~ N(0, sigma_log_size).

    Tying the galaxy to ``Rvir_acc`` rather than to ``Mstar`` is deliberate: it
    says a galaxy's size is set by its *halo*, so two galaxies of identical
    stellar mass can differ in size because they were accreted at different
    times. That extra scatter is inherited from the catalog for free -- at fixed
    M_V, ``Rvir_acc`` already carries ~0.17 dex of spread from the range of
    accretion redshifts, on top of the ``sigma_log_size`` we impose here.
    """
    rvir_acc = np.asarray(rvir_acc, dtype=float)
    r = size_norm * (rvir_acc / rvir_pivot) ** size_slope
    if sigma_log_size > 0:
        rng = np.random.default_rng(rng)
        r = r * 10.0 ** rng.normal(0.0, sigma_log_size, size=np.shape(r))
    return r


# Mean surface brightness within the half-light radius, for absolute magnitude
# M and half-light radius r in pc:
#
#     <mu>_e = M + 5 log10(r/pc) + C,
#     C = 2.5 log10(2 pi) - 5 + 5 log10(206265).
#
# The three terms are: spreading half the light over the area pi r^2 (hence the
# 2), the 10 pc zero point of the absolute-magnitude scale, and the conversion
# from radians to arcsec. C is distance-independent by construction -- the
# 5 log10(d) of the distance modulus cancels the -5 log10(d) of the angular
# size. Verified against the long way round (apparent magnitude at a real
# distance, angular size in arcsec) in notebook 03.
SB_CONST = 2.5 * np.log10(2.0 * np.pi) - 5.0 + 5.0 * np.log10(206264.806247)


def effective_surface_brightness(mv, rhalf):
    """
    Mean V surface brightness inside the half-light radius, mag arcsec^-2.

    ``rhalf`` in pc. Independent of distance, which is exactly why it is the
    right variable for a detectability cut on a *resolved* galaxy: a survey's
    limit is a limit on contrast against the sky, not on total flux.
    """
    mv = np.asarray(mv, dtype=float)
    rhalf = np.asarray(rhalf, dtype=float)
    return mv + 5.0 * np.log10(rhalf) + SB_CONST


# ---------------------------------------------------------------------------
# Convenience: run both over a satellite table
# ---------------------------------------------------------------------------
def add_observables(sats, mass_to_light_params=None, size_params=None, rng=None):
    """
    Add ``mv``, ``rhalf`` (pc) and ``mu_eff`` to a populated satellite table.

    Expects the columns ``mstar`` and ``rvir_acc``, i.e. the output of
    ``galaxy_halo.populate_subhalos``. Returns a copy.
    """
    rng = np.random.default_rng(rng)
    ml = dict(FIDUCIAL_MASS_TO_LIGHT, **(mass_to_light_params or {}))
    sz = dict(FIDUCIAL_SIZE, **(size_params or {}))

    out = sats.copy()
    out["mv"] = stellar_mass_to_mv(out["mstar"].to_numpy(), rng=rng, **ml)
    out["rhalf"] = assign_size(out["rvir_acc"].to_numpy(), rng=rng, **sz)
    out["mu_eff"] = effective_surface_brightness(out["mv"].to_numpy(),
                                                 out["rhalf"].to_numpy())
    return out
