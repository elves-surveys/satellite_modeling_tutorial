"""
Matching mock hosts to observed hosts.

A survey selects its hosts by *stellar* mass (or by K-band luminosity, or by
distance); the simulation is organized by *halo* mass. Comparing the two
requires a bridge, and no comparison can start without one:

    P(N_sat | Mstar_host) = INT P(N_sat | Mhalo) P(Mhalo | Mstar_host) dMhalo.

Following the plan, we use a **deterministic** mapping for Mhalo(Mstar_host) --
see ``host_stellar_to_halo_mass`` for what that costs.

Mass definition
---------------
Everything here is M200c: the mass inside the radius enclosing 200 times the
critical density, with h = 0.7. This is not a detail one may skip. The SatGen
runs used ``init.Rvir(M, Delta=200.)``, and the ELVES host table's ``Rvir``
column turns out to use the same convention -- ``virial_radius`` below
reproduces it to five significant figures for all 31 hosts. Had one side used
the Bryan & Norman virial overdensity (Delta ~ 101 rho_c at z = 0) instead, the
same halo would carry a radius 30% larger, and "one virial radius" would have
meant two different apertures.
"""

import numpy as np

__all__ = [
    "RHO_CRIT",
    "host_stellar_to_halo_mass",
    "virial_radius",
    "match_mock_hosts",
    "assign_host_halo_mass",
]

# Critical density at z = 0, Msun kpc^-3, for h = 0.7 (SatGen's cfg.rhoc0).
RHO_CRIT = 277.5 * 0.7 ** 2

# Rodriguez-Puebla et al. 2017 (arXiv:1703.04542), Table 6 / eq. 66, z = 0,
# in its inverted form: halo mass given stellar mass.
RP17_INVERSE = dict(logMh1=12.58, logMs0=10.90, beta=0.48, delta=0.29, gamma=1.52)


def host_stellar_to_halo_mass(logmstar, logMh1=12.58, logMs0=10.90, beta=0.48,
                              delta=0.29, gamma=1.52):
    """
    ``log10 M200c`` of a central galaxy's halo from its ``log10 Mstar``.

    The Rodriguez-Puebla et al. (2017) abundance-matching relation, inverted.
    This is the same relation the ELVES host table used to compute its ``Rvir``
    column, and the same one ``prep/`` used to attach a stellar mass to each
    SatGen host, so both sides of the comparison share one bridge.

    **It is deterministic, and the real relation is not.** At fixed host stellar
    mass, halo masses scatter by roughly 0.15-0.2 dex, and because the satellite
    abundance goes as N ~ Mhalo^1.1 that scatter propagates into a spread in
    N_sat that this mapping simply does not produce. Using it means the mock's
    host-to-host scatter is a *lower limit* on what the data should show. Adding
    the scatter is a natural extension; the plan calls the deterministic version
    acceptable for the tutorial, and it is, as long as one says so.
    """
    q = 10.0 ** (np.asarray(logmstar, dtype=float) - logMs0)
    return logMh1 + beta * np.log10(q) + q ** delta / (q ** (-gamma) + 1.0) - 0.5


def virial_radius(mvir, delta=200.0, rho_crit=RHO_CRIT):
    """R200c in kpc from M200c in Msun. Inverse of the usual definition."""
    mvir = np.asarray(mvir, dtype=float)
    return (3.0 * mvir / (4.0 * np.pi * delta * rho_crit)) ** (1.0 / 3.0)


def assign_host_halo_mass(obs_hosts, logmstar_col="logmstar"):
    """Add ``lgmvir`` (log10 M200c) and ``rvir_pred`` (kpc) to an observed host table."""
    out = obs_hosts.copy()
    out["lgmvir"] = host_stellar_to_halo_mass(out[logmstar_col].to_numpy())
    out["rvir_pred"] = virial_radius(10.0 ** out["lgmvir"].to_numpy())
    return out


def match_mock_hosts(sim_hosts, lgmvir, tolerance=0.1, mass_col="host_lgmvir"):
    """
    The simulated hosts within ``tolerance`` dex of ``lgmvir``.

    Returns the matching rows of ``sim_hosts``. The catalog is a 0.05 dex grid
    x 20 tree batches, so the default +-0.1 dex collects five grid points, i.e.
    of order 100 realizations -- enough to measure a mean, and enough to show
    the host-to-host scatter around it.

    Widening ``tolerance`` buys realizations at the price of blurring the host
    mass; since N_sat ~ Mhalo^1.1, +-0.2 dex is already a factor of 1.6 spread
    in the expected count, which will look like extra host-to-host scatter.
    """
    lgm = np.asarray(sim_hosts[mass_col], dtype=float)
    return sim_hosts[np.abs(lgm - float(lgmvir)) <= tolerance]
