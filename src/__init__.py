"""
Forward-modeling satellite populations -- KICP tutorial support code.

    io           load the preprocessed SatGen catalog and select satellites
    galaxy_halo  occupation and the stellar-to-halo mass relation
    observables  Mstar -> M_V, galaxy size, surface brightness
    projection   viewing geometry: external projection, and a Milky Way observer
    selection    survey selection and completeness, and the mock_observe API
    hosts        matching mock hosts to observed hosts
    statistics   cumulative counts, radial profiles, host-to-host scatter
    reference    literature SHMR / occupation curves, a few real MW dwarfs
    plotting     styling helpers
"""

from . import (galaxy_halo, hosts, io, observables, plotting,  # noqa: F401
               projection, reference, selection, statistics)

__all__ = ["io", "galaxy_halo", "observables", "projection", "selection",
           "hosts", "statistics", "reference", "plotting"]
