"""
Layer 3b of the forward model: where the observer stands.

Two branches, because the two datasets are geometrically different problems:

    project_satellites   external hosts (ELVES / ELVES-Dwarf). We sit far away,
                         see one random projection, and measure R_proj.
    place_mw_observer    the Milky Way. We sit *inside* the halo, 8.2 kpc from
                         its centre, and measure a heliocentric distance.

Both take the catalog's host-centric Cartesian ``x, y, z`` in kpc and add
columns; neither changes the intrinsic population.
"""

import numpy as np

__all__ = [
    "random_viewing_direction",
    "random_rotations",
    "project_satellites",
    "place_mw_observer",
    "R_SUN",
]

# Sun-Galactic-centre distance, kpc (GRAVITY Collaboration 2019: 8.178 kpc).
R_SUN = 8.2


def random_viewing_direction(size=1, rng=None):
    """
    ``size`` isotropically distributed unit vectors, shape ``(size, 3)``.

    Drawn as a normalized 3D Gaussian. Sampling ``theta`` uniformly instead is
    the classic mistake -- it piles points up at the poles.
    """
    rng = np.random.default_rng(rng)
    v = rng.normal(size=(int(size), 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def random_rotations(size=1, rng=None):
    """
    ``size`` random 3x3 rotation matrices, shape ``(size, 3, 3)``.

    Built by QR-decomposing a Gaussian matrix, which is the standard way to draw
    uniformly from O(3); we do not bother fixing the determinant to +1, because
    a reflection maps an isotropic distribution onto an isotropic distribution
    just as well as a rotation does.
    """
    rng = np.random.default_rng(rng)
    a = rng.normal(size=(int(size), 3, 3))
    q, r = np.linalg.qr(a)
    # QR is only unique up to the signs of R's diagonal; fixing them makes the
    # resulting Q uniform on the orthogonal group rather than merely orthogonal.
    return q * np.sign(np.einsum("nii->ni", r))[:, None, :]


def project_satellites(sats, direction=None, rng=None, host_col="host_id",
                       r_max_rvir=5.0, d_los_max=800.0):
    """
    View each host from a random direction and record projected quantities.

    Adds:

        ``r_proj_los``       projected host-centric radius, kpc
        ``d_los``            signed distance along the line of sight, kpc
        ``r_proj_over_rvir`` the same radius in units of the host's Rvir

    By default every host gets its **own** independent viewing direction, which
    is what a survey of many hosts actually is. Pass a single ``direction``
    (a 3-vector) to view them all the same way, e.g. to isolate the effect of
    changing the angle for one host.

    ``r_max_rvir`` drops objects beyond that many virial radii in 3D *before*
    projecting. A small tail of SatGen orbits goes numerically unbound (see
    ``prep/README.md``); without the guard, an object 200 Rvir away can land at
    small R_proj and masquerade as a satellite. Real surveys have this problem
    too -- it is called an interloper -- but it should come from real
    line-of-sight structure, not from the integrator. Pass ``None`` to disable.

    ``d_los_max`` then drops objects more than that many kpc from the host along
    the line of sight, in either direction. This stands in for the membership
    cut a real survey makes with distances rather than with velocities: ELVES
    confirms candidates with surface-brightness fluctuations, which at Local
    Volume distances separate a satellite from a background dwarf at roughly the
    Mpc level, so an object ~800 kpc in front of or behind the host would have
    been rejected. The cut is on the *depth* only, never on ``r_proj_los``, so it
    removes interlopers without touching the projected radial distribution of
    the genuine satellites. Pass ``None`` to keep everything and recover the
    uncorrected interloper fraction.

    Note the catalog's own ``r_proj`` column is a projection along the stored
    z axis. SatGen orbits are isotropic so that is a legitimate random
    projection, but it is a *fixed* one: it cannot show you how much the answer
    moves when you re-observe the same host from somewhere else. That is what
    this function is for.
    """
    out = sats.copy()
    if r_max_rvir is not None:
        out = out[out["r_3d"] < r_max_rvir * out["host_rvir"]].copy()

    xyz = out[["x", "y", "z"]].to_numpy(dtype=float)

    if direction is None:
        # one direction per host, mapped back onto the rows
        host_ids = out[host_col].to_numpy()
        uniq, inv = np.unique(host_ids, return_inverse=True)
        n_hat = random_viewing_direction(len(uniq), rng=rng)[inv]
    else:
        n_hat = np.asarray(direction, dtype=float).reshape(1, 3)
        n_hat = n_hat / np.linalg.norm(n_hat)
        n_hat = np.broadcast_to(n_hat, xyz.shape)

    d_los = np.einsum("ij,ij->i", xyz, n_hat)          # component along the LOS
    perp = xyz - d_los[:, None] * n_hat                 # component in the sky plane

    out["d_los"] = d_los
    out["r_proj_los"] = np.linalg.norm(perp, axis=1)
    out["r_proj_over_rvir"] = out["r_proj_los"] / out["host_rvir"]

    if d_los_max is not None:
        out = out[np.abs(out["d_los"]) < d_los_max].copy()
    return out


def place_mw_observer(sats, r_sun=R_SUN, direction=None, rng=None,
                     host_col="host_id"):
    """
    Put an observer inside the host, ``r_sun`` kpc from its centre, and compute
    heliocentric distances and sky positions.

    Adds:

        ``d_helio``    distance from the observer, kpc
        ``dm``         distance modulus, ``5 log10(d/10 pc)``
        ``cos_theta``  cosine of the angle between the satellite and the
                       Galactic centre as seen from the observer
        ``ra, dec``    a position on the real sky, degrees -- see below

    ``direction`` is the unit vector from the host centre to the observer;
    random if not given. These SatGen runs have ``fd = fb = 0`` -- no disk, no
    bulge -- so the halo has no preferred plane and there is no physically
    meaningful place to put the Sun. Random is the honest choice, and it means
    ``cos_theta`` carries no information about a Galactic-plane mask.

    **What ``ra`` and ``dec`` are, and are not.** Each host's sky is rotated by
    its own random rotation before being written down in equatorial coordinates.
    They are therefore *not* predictions of where these satellites would be --
    the simulation has no disk to align to the real Galactic plane, so no such
    prediction exists. What they are is a way to ask the real survey masks a
    question they can answer: given a satellite population that is isotropic
    with respect to the footprint, how many land in the searched sky? Averaged
    over many realizations that is the same answer as multiplying by the
    footprint area fraction -- but a *single* realization sees a footprint with
    holes in it, and only this version gets that scatter right.

    One rotation per host, not one per satellite, because satellites of a host
    are not independent points on the sky, and pretending they are would
    understate exactly the scatter this exists to capture.
    """
    rng = np.random.default_rng(rng)
    out = sats.copy()
    if direction is None:
        direction = random_viewing_direction(1, rng=rng)[0]
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    sun = r_sun * direction
    rel = out[["x", "y", "z"]].to_numpy(dtype=float) - sun   # observer -> satellite
    d = np.linalg.norm(rel, axis=1)

    out["d_helio"] = d
    out["dm"] = 5.0 * np.log10(np.maximum(d, 1e-6) * 1e3 / 10.0)
    # the Galactic centre lies along -direction as seen from the observer
    with np.errstate(invalid="ignore", divide="ignore"):
        out["cos_theta"] = (rel @ (-direction)) / d

    u = rel / np.maximum(d, 1e-12)[:, None]
    if host_col in out.columns:
        _, inv = np.unique(out[host_col].to_numpy(), return_inverse=True)
        rot = random_rotations(inv.max() + 1 if len(inv) else 1, rng=rng)[inv]
    else:
        rot = np.broadcast_to(random_rotations(1, rng=rng), (len(out), 3, 3))
    v = np.einsum("nij,nj->ni", rot, u)

    out["ra"] = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    out["dec"] = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    return out
