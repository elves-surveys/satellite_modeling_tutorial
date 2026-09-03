"""
Loading the preprocessed SatGen catalog.

The catalog was built by ``prep/build_catalog.py``; see ``prep/README.md`` for
the SatGen conventions, the resolution limit, and why orphans are kept.

Nothing in this module depends on SatGen itself -- participants only ever see
the two parquet tables.
"""

import os

import numpy as np
import pandas as pd

__all__ = [
    "DATA_DIR",
    "DEFAULT_PREFIX",
    "OBS_DIR",
    "load_hosts",
    "load_subhalos",
    "select_satellites",
    "load_elves_hosts",
    "load_elves_satellites",
    "load_mw_satellites",
    "load_census_sims",
    "load_census_galaxies",
    "load_census_chain",
    "load_census_footprint",
]

# ``KICP_tutorial/data`` is a symlink to the scratch directory holding the
# parquet tables. Override with the KICP_DATA_DIR environment variable.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("KICP_DATA_DIR", os.path.join(_PKG_ROOT, "data"))

# The observational comparison tables. Unlike the SatGen parquet files these are
# small enough to live in the repository, so they sit in their own directory
# rather than behind the machine-specific ``data`` symlink. Rebuild them with
# ``prep/build_obs_tables.py``.
OBS_DIR = os.environ.get("KICP_OBS_DIR", os.path.join(_PKG_ROOT, "data_obs"))


# Which catalog the loaders read by default. The parquet tables are named
# ``<prefix>_hosts.parquet`` / ``<prefix>_subhalos.parquet``, one prefix per
# SatGen run (see ``prep/build_catalog.py``'s RUN_DEFAULTS) or per trimmed
# subset (``prep/trim_catalog.py``):
#
#   "satgen"          zzli_dwarf, the tutorial default -- SatEvo, Mres=1e7,
#                     log Mvir 10.50-12.00, R_vir = R_200c
#   "satgen_new"      subevo_green, the paper-2 SubEvo/Green run, log Mvir
#                     10.50-12.50 with R_vir = Bryan & Norman
#   "satgen_new_MW"   subevo_green trimmed to log Mvir >= 11.5 (284 hosts)
#   "satgen_sd"       sdanieli, coarser Mres=1e8 but reaching cluster scale
#
# The runs are *not* interchangeable: they use different subhalo profiles and
# different virial-mass definitions. See ``prep/README.md``.
DEFAULT_PREFIX = os.environ.get("KICP_CATALOG_PREFIX", "satgen")

# Padding for float64 thresholds compared against the float32 `host_lgmvir`
# stored in the subhalo table. See `select_satellites`.
_LGM_ATOL = 1e-4


def load_hosts(data_dir=None, prefix=DEFAULT_PREFIX):
    """
    One row per SatGen host realization.

    ``prefix`` selects which catalog to read; see ``DEFAULT_PREFIX``. The default
    ``zzli_dwarf`` catalog has 619 hosts on a 0.05 dex grid in log Mvir x 20 tree
    batches.
    """
    data_dir = data_dir or DATA_DIR
    return pd.read_parquet(os.path.join(data_dir, f"{prefix}_hosts.parquet"))


def load_subhalos(data_dir=None, full=False, columns=None, prefix=DEFAULT_PREFIX):
    """
    One row per subhalo, ``Mpeak > 1e7`` = SatGen's own resolution limit.

    By default this reads the slim table, from which objects that merged into
    the host centre have already been dropped. ``full=True`` reads the 227 MB
    audit table that still contains them -- note that trimmed catalogs built by
    ``prep/trim_catalog.py`` only carry a ``_full`` table if it was asked for.

    ``prefix`` selects the catalog; see ``DEFAULT_PREFIX``.
    """
    data_dir = data_dir or DATA_DIR
    kind = "subhalos_full" if full else "subhalos"
    return pd.read_parquet(os.path.join(data_dir, f"{prefix}_{kind}.parquet"),
                           columns=columns)


def select_satellites(subs, mpeak_min=1e8, aperture="r_proj", f_rvir=1.0,
                      keep_orphans=True, keep_merged=False,
                      host_lgmvir_range=None, host_ids=None):
    """
    Apply the selections that define "a subhalo we are willing to call a satellite".

    The defaults encode the treatment recommended in ``prep/README.md``:

    1. cut on ``mpeak``, never on the present-day mass ``mass_now`` -- a galaxy's
       stars sit deep inside its halo, so stripping 99% of the dark matter does
       not remove the galaxy;
    2. keep orphans (stripped below SatGen's numerical mass floor but still
       orbiting) -- dropping them would delete ~90% of the surviving population
       just above the floor, and would act as an implicit, strongly
       Mpeak-dependent disruption model;
    3. always apply an aperture -- a small tail of orbits goes numerically
       unbound, and an orphan's radius should never be used bare.

    ``aperture`` is the column to cut on: ``"r_proj"`` (projected, ELVES-like)
    or ``"r_3d"``. Pass ``None`` to skip the cut.

    ``host_ids`` restricts to an explicit list of hosts. Prefer it over
    ``host_lgmvir_range`` whenever you also need the matching host list --
    see ``host_bin``.
    """
    mask = subs["mpeak"] > mpeak_min

    # The slim catalog has already had merged objects removed, so it carries no
    # `is_merged` column; only the full audit table does.
    if not keep_merged and "is_merged" in subs.columns:
        mask &= ~subs["is_merged"]
    if not keep_orphans:
        mask &= ~subs["is_orphan"]
    if aperture is not None:
        mask &= subs[aperture] < f_rvir * subs["host_rvir"]
    if host_lgmvir_range is not None:
        # `host_lgmvir` is stored float32 in the subhalo table and float64 in the
        # host table, so the *same* grid mass compares differently against a
        # float64 threshold: float32(11.90) = 11.899999618... < 11.9, while
        # float64(11.90) = 11.900000000... >= 11.9. Without the padding, a bin
        # edge that lands on a grid mass keeps those hosts in the host list but
        # drops every one of their satellites, and they are then counted as
        # genuine zeros. The grid is 0.01 dex at its finest and the float32 error
        # near log M = 12 is ~1e-6 dex, so 1e-4 separates the two cleanly.
        lo, hi = host_lgmvir_range
        mask &= ((subs["host_lgmvir"] >= lo - _LGM_ATOL)
                 & (subs["host_lgmvir"] <= hi + _LGM_ATOL))
    if host_ids is not None:
        mask &= subs["host_id"].isin(np.asarray(host_ids))

    return subs[mask].copy()


def host_bin(subs, hosts, lgmvir, width=0.1, **kwargs):
    """
    Satellites of every host within ``+/- width`` of ``lgmvir``, plus the ids of
    all hosts in that bin.

    Returning the host ids matters: hosts with zero satellites above the cut
    leave no rows in ``subs``, and forgetting them biases every per-host
    statistic high. See ``statistics.per_host_counts``.

    The bin is defined once, on the host table, and the satellites are then
    selected by ``host_id``. Do not instead apply the same mass cut to both
    tables: they store ``host_lgmvir`` at different precision, so a bin edge
    landing on a grid mass selects different hosts on each side, and every
    satellite of an edge host disappears while the host itself stays in the
    list -- which `per_host_counts` then reads as a true zero.
    """
    lo, hi = lgmvir - width, lgmvir + width
    in_bin = hosts[(hosts["host_lgmvir"] >= lo) & (hosts["host_lgmvir"] <= hi)]
    host_ids = in_bin["host_id"].to_numpy()
    return select_satellites(subs, host_ids=host_ids, **kwargs), host_ids


# ---------------------------------------------------------------------------
# Observational comparison samples
# ---------------------------------------------------------------------------
def load_elves_hosts(obs_dir=None, require_coverage=True):
    """
    The ELVES hosts (Carlsten et al. 2022): 31 nearby Local Volume galaxies.

    Columns: ``name``, ``dist`` (Mpc), ``mv_host``, ``logmstar``, ``r_cover``
    (the projected radius searched, kpc), ``rvir`` (kpc, R200c from the
    Rodriguez-Puebla+17 relation) and ``r_search = min(r_cover, rvir)``.

    ``require_coverage=True`` drops hosts with ``r_cover = 0``. There is one
    (NGC 3621), and it is *not* a host with zero satellites -- it is a host that
    was never searched. The two are opposite: dropping a genuine zero biases
    N_sat high, keeping an unsearched host biases it low. Only the second
    applies here.
    """
    df = pd.read_csv(os.path.join(obs_dir or OBS_DIR, "elves_hosts.csv"))
    return df[df["r_cover"] > 0].reset_index(drop=True) if require_coverage else df


def load_elves_satellites(obs_dir=None, psat_min=1.0, drop_local_group=False):
    """
    The ELVES satellite candidates.

    ``psat`` is the published probability that a candidate really is a satellite
    of its host, from a surface-brightness-fluctuation and morphology model.
    ``psat_min=1.0`` keeps only the secure ones, which is the conservative choice
    and the one the ELVES-Dwarf notebooks use; pass ``psat_min=0`` and weight by
    ``psat`` instead to keep the marginal objects. This is the P_membership term
    that the mock does not model, so whichever way you cut it, the mock is
    comparing against a sample that has had a step applied to it that the mock
    has not.

    ``drop_local_group=True`` removes the Milky Way and M31 satellites, which
    ELVES takes from McConnachie (2012) rather than from its own imaging. They
    reach far fainter than the rest of the survey, so including them mixes two
    very different selection functions in one sample.
    """
    df = pd.read_csv(os.path.join(obs_dir or OBS_DIR, "elves_satellites.csv"))
    if psat_min is not None:
        df = df[df["psat"] >= psat_min]
    if drop_local_group:
        df = df[~df["from_local_group"]]
    return df.reset_index(drop=True)


def load_mw_satellites(obs_dir=None, confirmed_only=True, galaxies_only=False):
    """
    The known Milky Way dwarf satellites, from the Local Volume Database
    (Pace 2024): ``mv``, ``rhalf`` (pc, circularized), ``d_helio`` (kpc),
    ``logmstar``, ``mu_eff``.

    ``confirmed_only`` keeps objects confirmed to be real detections.
    ``galaxies_only`` additionally requires ``confirmed_galaxy``, which excludes
    the objects still argued over as star clusters -- a real ambiguity at the
    faint end, and one more reason the observed count is not a single number.
    """
    df = pd.read_csv(os.path.join(obs_dir or OBS_DIR, "mw_satellites.csv"))
    if confirmed_only:
        df = df[df["confirmed_real"] == 1]
    if galaxies_only:
        df = df[df["confirmed_galaxy"] == 1]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The DELVE Milky Way Census release (Tan & Drlica-Wagner et al. 2025)
# ---------------------------------------------------------------------------
# Built by ``prep/build_census_tables.py`` from
# https://github.com/delve-survey/delve_mw_census (doi:10.5281/zenodo.18383157).


def load_census_sims(obs_dir=None, survey=None):
    """
    The injection-recovery simulations behind the census selection function.

    One row per simulated satellite injected into the real DES Y6 or DELVE DR3
    catalogs and passed through the real search pipelines: ``mv``, ``rhalf`` (pc,
    circularized), ``d_helio`` (kpc), ``density`` (foreground stars/arcmin^2)
    and ``detected``.

    This *is* the selection function, before anyone fit a curve to it. Trimmed
    to the Tan+25 box, it is 59,150 rows over two surveys.

    ``density`` is populated for DES Y6 only; the DELVE DR3 table leaves it at
    zero for most rows, so do not use it to split that survey.
    """
    df = pd.read_parquet(os.path.join(obs_dir or OBS_DIR, "census_sims.parquet"))
    if survey is not None:
        df = df[df["survey"] == survey]
    return df.reset_index(drop=True)


def load_census_galaxies(obs_dir=None, in_census=True, in_box=True,
                         classes=("D", "PD")):
    """
    The census reference satellites -- the sample Tan+25 measure against.

    ``in_census`` keeps the objects inside the census footprint, which are the
    ones the selection function describes; ``in_box`` keeps those inside
    ``reference.TAN25_BOX``. Both default to True, giving the **51 satellites**
    that are the correct comparison for a mock observed catalog.

    ``classes`` filters ``dwarf_class``: D (dwarf), PD (probable dwarf),
    A (ambiguous). The default keeps D and PD and drops the single ambiguous
    object, matching the paper.

    Note this is *not* ``load_mw_satellites``. The LVDB sample is everything
    known, including objects found in data deeper than the census; this one is
    the census's own sample, with a ``survey`` column saying which survey covers
    each object. Comparing a census-selected mock against the LVDB sample mixes
    two selection functions.
    """
    df = pd.read_csv(os.path.join(obs_dir or OBS_DIR, "census_galaxies.csv"))
    if in_census:
        df = df[df["in_census"]]
    if in_box:
        df = df[df["in_box"]]
    if classes is not None:
        df = df[df["dwarf_class"].isin(classes)]
    return df.reset_index(drop=True)


def load_census_footprint(obs_dir=None):
    """
    The census footprint: what fraction of each patch of sky each survey searched.

    Returns a dict with

        ``surveys``       ``('des_y6', 'delve_dr3', 'ps1_dr1')``
        ``coverage``      ``(3, 180, 360)``, the fraction of each cell that
                          survey usably covers, in the same order as ``surveys``
        ``ra_edges``      ``(361,)`` degrees
        ``sindec_edges``  ``(181,)`` **sin(Dec)**, not Dec

    Binning uniformly in sin(Dec) is what makes every one of the 64,800 cells the
    same 0.637 deg^2, so a coverage fraction can be averaged and summed with no
    area weights, and indexing costs two divisions. ``selection.census_sky_weights``
    does the lookup; ``plotting.plot_sky_map`` draws it.

    This is ``prep/build_census_tables.py``'s reduction of the release's three
    nside = 4096 HEALPix masks (24 MB) down to 55 kB. It reproduces each survey's
    published area to 0.1%. What it throws away is structure below ~1 degree --
    the sub-arcminute holes around bright stars and around already-known dwarfs.
    Do not use it to ask whether one particular object was masked; use it to ask
    what fraction of the sky was searched, which is the only question a
    population comparison can ask anyway.
    """
    with np.load(os.path.join(obs_dir or OBS_DIR, "census_footprint.npz")) as f:
        return dict(surveys=tuple(str(s) for s in f["surveys"]),
                    coverage=f["coverage"].astype(float) / 255.0,
                    ra_edges=f["ra_edges"], sindec_edges=f["sindec_edges"])


def load_census_chain(obs_dir=None):
    """
    The thinned posterior of the Tan+25 empirical model, 4,000 x 6.

    Columns ``[beta, rcore, sigma_sl, zp, sp, n_total]``; see
    ``reference.tan25_luminosity_function``.
    """
    return np.load(os.path.join(obs_dir or OBS_DIR, "census_chain.npy"))
