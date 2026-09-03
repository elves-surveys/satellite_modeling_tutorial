# `src/` — the forward-model code

Standalone support code for the tutorial. It depends only on numpy, scipy, pandas,
matplotlib and pyarrow — **not** on the ELVES-Dwarf repo — so the tutorial can be
distributed as `src/ + notebooks/ + data_obs/ + the two parquet files`.

| module | contents |
|---|---|
| `io.py` | `load_hosts`, `load_subhalos`, `select_satellites`, `host_bin`, `load_census_*` |
| `galaxy_halo.py` | `occupation_fraction`, `assign_occupation`, `mean_stellar_mass`, `assign_stellar_mass`, `populate_subhalos` |
| `observables.py` | `stellar_mass_to_mv`, `assign_size`, `effective_surface_brightness`, `add_observables` |
| `projection.py` | `random_viewing_direction`, `random_rotations`, `project_satellites`, `place_mw_observer` |
| `selection.py` | `apply_aperture`, `detection_probability_mstar`, `mw_detection_probability`, `census_sky_weights`, `census_detection_probability`, `apply_completeness`, `mock_observe` |
| `hosts.py` | `host_stellar_to_halo_mass`, `virial_radius`, `match_mock_hosts` |
| `statistics.py` | `cumulative_counts`, `per_host_counts`, `percentile_band`, `radial_profile`, `stacked_profile`, `scatter_budget` |
| `reference.py` | literature curves: Nadler+20, Danieli+23, Dooley+16; the published MW census selection function (Drlica-Wagner+20, Tan+25); a few real MW dwarfs |
| `plotting.py` | `use_tutorial_style`, `plot_band`, `sky_to_mollweide`, `plot_sky_coverage` |

The SatGen parquet tables come from `KICP_tutorial/data` (a symlink into `/scratch`),
overridable with `KICP_DATA_DIR`. The three observational CSVs are small enough to be
committed and live in `KICP_tutorial/data_obs`, overridable with `KICP_OBS_DIR`; rebuild
them with `prep/build_obs_tables.py`.

---

## Fiducial parameters, and where they came from

```python
FIDUCIAL_OCCUPATION   = dict(logM50=8.0, sigma_gal=0.4)
FIDUCIAL_SHMR         = dict(alpha=2.0, logM0=10.0, logMstar0=6.5, sigma_logMstar=0.2)
FIDUCIAL_MASS_TO_LIGHT = dict(mass_to_light_v=1.5, sigma_log_ml=0.0)
FIDUCIAL_SIZE          = dict(size_norm=1500.0, size_slope=1.0,
                              sigma_log_size=0.2, rvir_pivot=100.0)
```

**The SHMR is not a toy.** Fitting a straight line to the published relations over
$M_{\rm peak}=10^8-10^{11}$, with the pivot at $10^{10}$:

| relation | $\alpha$ | $\log M_{\star,0}$ | rms residual |
|---|---|---|---|
| Nadler+20 | 1.977 | 6.476 | 0.064 dex |
| Danieli+23 | 2.098 | 6.488 | **0.002 dex** |

So the transparent power law *is* the literature relation over the range that matters.
The two published relations still differ by 0.27 dex at $10^8 M_\odot$ — a real
disagreement, not a fitting artifact.

**The occupation threshold is the weakest link.** Nadler+20 infer $M_{50}\sim10^{7.5}$
from the MW satellites; the Dooley+16 reionization curve (`reference.dooley16_occupation`,
originally Barber+14) is a much sharper cutoff, well fit by an erf with
`logM50=8.752, sigma_gal=0.333`. Those disagree by a **factor of 18 in halo mass**. The
fiducial 8.0 sits between them; treat it as a knob, not a measurement.

---

## Two things that are easy to get wrong

**1. Count hosts with zero satellites.** `per_host_counts` takes an explicit `host_ids`
argument for this reason. A host with no satellites above a threshold contributes no rows
to the table, so deriving the host list from the data alone silently drops it and biases
the median and percentile band high — worst exactly at the bright end, where counts are
small and the science is.

**2. Use the mean, not the median, for abundance ratios.** A median of small integer
counts is quantized (2, 3, 4, ...), which turns a smooth ratio into a staircase. The
median is the right choice for showing host-to-host scatter; the mean is the right choice
for comparing population abundances between models.

---

## Selection defaults

`select_satellites` defaults to `mpeak_min=1e8, aperture="r_proj", f_rvir=1.0,
keep_orphans=True, keep_merged=False`, following `prep/README.md`:

1. cut on `mpeak`, never on `mass_now`;
2. keep orphans, drop merged objects;
3. always apply an aperture.

Note that `select_satellites` tolerates a missing `is_merged` column — the slim catalog
has already had merged objects removed, so only the full audit table carries the flag.

---

## The observable layer (`observables.py`, `projection.py`)

**Units, fixed once.** `Mstar` in Msun, `M_V` absolute Vega magnitude, `rhalf` in **pc**,
`rvir_acc` in **kpc** (as in the catalog), `mu_eff` in mag arcsec^-2.

**The size relation is not a toy either.** `size_norm=1500` pc at a pivot of `rvir_pivot=100`
kpc with unit slope is exactly $r_{1/2}=0.015\,R_{\rm vir}$, i.e. Kravtsov (2013). Checked
against `reference.mw_dwarfs()` in notebook 03: the mock median tracks the real dwarfs to within
a factor of ~1.5 from $M_V=-14$ to $-3$, with nothing fitted.

Hanging the size on $R_{\rm vir,acc}$ rather than on $M_\star$ has a payoff: at fixed $M_V$ the
catalog already carries ~0.16 dex of spread in $R_{\rm vir,acc}$ from the range of accretion
redshifts, so ~40% of the size *variance* at fixed luminosity is predicted rather than imposed.

**The surface-brightness constant is derived, not quoted.**

```python
SB_CONST = 2.5*np.log10(2*np.pi) - 5.0 + 5.0*np.log10(206264.806247)   # = 23.5676
```

for `rhalf` in pc. Notebook 03 checks it against the long way round (apparent magnitude at a
real distance, angular size in arcsec) and asserts agreement to 1e-9, and confirms the result is
distance-independent.

**`reference.MW_DWARFS` is not a sample.** 22 familiar objects spanning $M_V=-18$ to $-1.5$,
for checking that the size model lands in the right part of the plane. It is incomplete and
inhomogeneous — never count rows and compare with a mock.

---

## Three more things that are easy to get wrong

**3. Magnitudes count the other way.** `cumulative_counts` and `per_host_counts` take
`direction="below"` for exactly this: $N(<M_V)$ is the same statistic as $N(>M_\star)$.

**4. Never normalize a radial profile per host and take the median.** A host with two
satellites has a normalized profile that can only step $0\to0.5\to1$, so the median over hosts
is biased strongly toward a concentrated profile — badly enough to invent a spurious
universality across host mass. Use `stacked_profile`, which pools satellites first and
bootstraps over *hosts* (satellites of one host are not independent).
`radial_profile(normalize=True)` still exists, for the genuine scatter question.

**5. Guard the projection.** `project_satellites` drops objects beyond `r_max_rvir=5` in 3D
before projecting. A handful of SatGen orbits go numerically unbound; without the guard an
object 200 Rvir away can land at small $R_{\rm proj}$ and masquerade as a satellite. Real
interlopers should come from real structure, not from the integrator.

---

## Numbers worth remembering from notebook 03

| quantity | value |
|---|---|
| $N_{\rm sat}(>10^5M_\odot)$ at $\log M_{\rm vir}=12$ | $20.2 \pm 8.8$ per host |
| ... of which shot noise | $\sqrt{20.2}=4.5$; the rest, 7.6, is real assembly variance |
| projected vs 3D aperture | 1.11x the counts, 10% of them interlopers |
| scatter from the viewing angle | ~6%, against ~44% host-to-host |
| doubling the size normalization | leaves the intrinsic population identical, halves the count above a surface-brightness limit |

---

## The selection layer (`selection.py`, `hosts.py`)

| module | contents |
|---|---|
| `selection.py` | `apply_aperture`, `detection_probability_mstar`, `detection_probability_mv`, `n_resolved_stars`, `mw_detection_probability`, `census_detection_probability`, `apply_completeness`, `apply_survey_selection`, `mock_observe`, `ELVES_SURVEY`, `MW_SURVEY`, `MW_SURVEY_TOY` |
| `hosts.py` | `host_stellar_to_halo_mass`, `virial_radius`, `assign_host_halo_mass`, `match_mock_hosts` |
| `io.py` | added `load_elves_hosts`, `load_elves_satellites`, `load_mw_satellites`, `OBS_DIR` |

A survey is a dictionary, and `mock_observe` uses it **as given** rather than merging it with a
default, so a survey is always described in one visible place:

```python
ELVES_SURVEY = dict(geometry="projected",
                    aperture=dict(f_rvir=1.0),
                    selection=dict(kind="mv", mv50=-9.0, sigma_det=0.3, p_max=0.9),
                    requires_mv=True)

MW_SURVEY    = dict(geometry="mw",
                    aperture=dict(r_max_kpc=400.0),
                    selection=dict(kind="census", width=0.5))
```

`mock_observe(..., detected_only=False)` returns the full post-aperture table with `p_det` and
`detected` columns. Use that for any intrinsic-versus-observed comparison: both populations then
come from the *same* draw, so the difference between the curves is the selection function and
nothing else.

`mock_observe` also applies `mpeak_min=1e8` by default, matching `select_satellites`. The
catalog reaches 10^7, but both published SHMRs stop at 10^8, and extrapolating a slope-2 power
law the rest of the way predicts galaxies of a few solar masses.

### For the Milky Way, use the published selection function

`reference.MW_CENSUS_P50` carries the analytic 50% detectability contours from
**Drlica-Wagner et al. 2020** (arXiv:1912.03302; DES Y3 and PS1 DR1 — the selection function
behind Nadler+20, whose SHMR this tutorial already uses) and **Tan et al. 2025**
(arXiv:2509.12313; DES Y6 and DELVE DR3). Both derive it by injecting tens of thousands of
simulated satellites into the real survey catalogs and running the real search pipelines. Both
write it the same way, at fixed heliocentric distance:

$$\log_{10}(r_{1/2}/{\rm pc}) = \frac{A_0(D)}{M_V - M_{V,0}(D)} + \log_{10}r_{1/2,0}(D)$$

— three numbers per distance per survey. `reference.census_p50_mv` solves it for $M_{V,50}$ and
interpolates the parameters in $\log D$; `selection.census_detection_probability` softens it into
a logistic and averages over the de-overlapped census footprint (DES Y6 4,869 deg², DELVE DR3
12,006 deg², PS1 DR1 10,851 deg², and a third of the sky with no census at all, so $P_{\rm det}$
saturates at 0.67). `MW_SURVEY` uses it; `MW_SURVEY_ISOTROPIC` is the same thing with the sky
mask and the bright-end assumption switched off, and `MW_SURVEY_TOY` keeps the from-scratch
version.

Three properties, all checked in notebook 05:

- **the softening width is measured, not assumed** — see the next section. Between 0.25 and
  1.0 mag the predicted count moves by one satellite either way;
- **the footprint average assumes isotropy**, which is right for a mock with a random observer
  and wrong for the known dwarfs — applied to them it predicts 39 census recoveries against the
  49 Tan et al. report, because the real satellites crowd into the deep southern sky;
- **the size dependence is enormous.** At $D=100$ kpc, DELVE DR3 reaches $M_V=-3.1$ for a 30 pc
  satellite and only $-7.3$ for a 1000 pc one. Four magnitudes, same distance, same survey.

`reference.TAN25_POPULATION` and `TAN25_SIZE_LUMINOSITY` carry their results for comparison:
265 (+79/−47) satellites in a stated box, 49 recovered, and the completeness-corrected
size–luminosity relation $\langle\log_{10}r_{1/2}\rangle = 2.07 - 0.12(M_V+6)$ with 0.24 dex
scatter. **Compare a mock's intrinsic sizes to that relation, not to the observed dwarfs** —
compact satellites are easier to find, so the detected population is biased compact, and the
naive comparison makes a correct size model look wrong by a factor of three.

### The census data release: measuring the width, and the right comparison sample

`prep/build_census_tables.py` distils the
[DELVE MW Census release](https://github.com/delve-survey/delve_mw_census)
(doi:10.5281/zenodo.18383157) into four committed files, 1.13 MB, no new dependencies at tutorial
run time. Notebook 05 is built on them.

| loader | contents |
|---|---|
| `io.load_census_sims()` | 59,150 injected satellites with `detected` — the selection function before anyone fit a curve to it |
| `io.load_census_galaxies()` | the **51** census satellites inside `reference.TAN25_BOX` |
| `io.load_census_chain()` | 4,000 posterior samples: `[beta, rcore, sigma_sl, zp, sp, n_total]` |
| `io.load_census_footprint()` | the searched sky, 64,800 equal-area cells x 3 surveys, 55 kB |

**`reference.CENSUS_WIDTH` is fitted to the simulations**: 0.32 mag (DES Y6),
0.38 (DELVE DR3), with PS1 borrowing DELVE's because its simulations belong to Drlica-Wagner+20.
`census_detection_probability(width=None)` — the default, and what `MW_SURVEY` uses — takes them
per survey. Two checks: the recovery fraction within 0.15 mag of the published contour is 0.494 /
0.510, so it really is the 50% contour; and one global width reproduces the simulations' own
recovered counts to 0.3%, and to 4% in every octave of distance.

**Use `load_census_galaxies()`, not `load_mw_satellites()`, for anything census-selected.** LVDB
is everything known, including objects found in data deeper than the census; mixing the two mixes
two selection functions. The concrete difference: Eridanus II is at $D_{\rm GC}=372$ kpc, outside
Tan+25's 300 kpc box, so the $-8\le M_V<-6$ bin holds one satellite, not two.

### Three ways to handle the sky, and what each is worth

`census_detection_probability` takes the footprint three ways. With `survey="des_y6"` it gives
one field's sensitivity everywhere — for comparing surveys, not for predicting counts. By
default it averages over the footprint by area, which is exactly right for the *mean* of an
isotropic mock. Given `ra=`/`dec=` it looks each object up in the real mask through
`census_sky_weights`, which `projection.place_mw_observer` makes possible by giving every host
its own random orientation on the sky. Measured on 330 mock Milky Ways in notebook 05: the mask
moves the mean by 0.3% and the **scatter by a third**. Use it when the question is whether one
observed number is consistent with a model, since that question is only about the width of the
predicted distribution.

`bright_all_sky_mv` (default `reference.CENSUS_BRIGHT_ALL_SKY_MV = -8`, used by `MW_SURVEY`)
sets $P_{\rm det}=1$ for the classical dwarfs, which were catalogued all-sky long before DES;
without it the model gives Fornax and Sculptor $P_{\rm det}=0.67$ and under-predicts the bright
end by that factor. It is a judgement call — Tan+25's own code uses $-12.5$ — and it obliges you
to add the bright satellites found *outside* the footprint to the observed sample. There is
exactly one, Antlia II.

**`reference.tan25_luminosity_function(mv, chain)` is the completeness-corrected LF**, and
`tan25_radial_profile(d_gc, chain)` the completeness-corrected radial distribution (their cored
$\mathrm{d}N/\mathrm{d}D \propto D^2(D+r_{\rm core})^{-3}$), both with the full posterior.
Compare them to a mock's *intrinsic* population; compare the census sample to the mock's
*detected* one. Doing only the second, at a single threshold, is how a comparison agrees to a
few percent while being wrong: bin by bin the mock is high by 1.4–1.9× wherever the census
actually constrains anything, and the totals agree only because **54% of Tan+25's headline 265
lies fainter than $M_V=-2$**, where they detect almost nothing.

The trained classifiers (~4.5 MB) are deliberately unused: they need `xgboost`, and what they add
beyond the analytic contour is the foreground stellar density within a field, worth ~0.2 mag
against a 0.35 mag softening width. The HEALPix masks *are* used, reduced to
`load_census_footprint()` above; they reproduce the published areas to 0.1%
(4869 / 12006 / 10851 deg²). `healpy` is needed to rebuild that file in `prep/`, never to read
it.

### Why the contour has the shape it does (and the toy that shows it)

The obvious model for a resolved-star search is a surface-brightness threshold that degrades
with distance. The Local Volume Database says that is the wrong shape. Binning the 63 confirmed
Milky Way dwarfs by distance:

| $D$ (kpc) | faintest $\langle\mu_V\rangle_e$ found | faintest $M_V$ found |
|---|---|---|
| 0–50 | 31.1 | 0.0 |
| 50–100 | 30.9 | −0.9 |
| 100–175 | 30.7 | −3.1 |
| 175–400 | 30.4 | −4.2 |

The surface-brightness limit barely moves (0.7 mag over a factor of ten in distance, as it must,
since $\langle\mu\rangle_e$ is distance independent); the **luminosity** limit moves by four
magnitudes. So the distance dependence cannot live in a surface-brightness threshold. It lives
in the number of member stars above the survey's magnitude limit, which is what
`n_resolved_stars` computes. `mw_detection_probability` multiplies the two:

$$P_{\rm det} = f_{\rm sky}\times \sigma\!\left(\tfrac{\log N_\star - \log N_{\rm min}}{w_N}\right)
\times \sigma\!\left(\tfrac{\mu_{\rm lim} - \langle\mu_V\rangle_e}{w_\mu}\right)$$

The star-count model, `n(<M)/L_V = 10^{0.45(M-6.5)}` capped at 1, is calibrated on two real
objects rather than an isochrone: Boötes I (~40 members brighter than $M_V\approx0.4$) and
Segue 1 (~70 brighter than $M_V\approx4.9$). It reproduces both to 30%, and its 50% locus tracks
the observed $M_V$–$D$ envelope to half a magnitude with nothing fitted to it.

**It is a toy, and it is superseded** by `census_detection_probability` above — its thresholds
were placed against the faint envelope of the *known* dwarfs, which is circular. It is kept
because it explains, from first principles, why the published contour bends the way it does; the
notebook builds it first and then looks up the answer.

---

## Three more things that are easy to get wrong (selection)

**6. Check the mass definition before matching hosts.** SatGen's `TreeGen` was called with
`Delta=200.`, and the ELVES host table's `Rvir` column turns out to use the same convention —
`hosts.virial_radius` reproduces it to six significant figures for all 30 hosts. That was worth
checking rather than assuming: the Bryan & Norman virial overdensity ($\Delta_{\rm vir}\approx101
\rho_{\rm crit}$ at $z=0$) attaches a radius **26% larger** to the same mass, and "one virial
radius" would then have meant two different apertures on the two sides of the comparison.

**7. An unsearched host is not a host with zero satellites.** `load_elves_hosts` drops
NGC 3621, whose `r_cover` is 0. Dropping a genuine zero biases $N_{\rm sat}$ high (see point 1);
keeping an unsearched host biases it low. Only the second applies.

**8. Apply a survey's completeness in the variable the survey measured it in.** ELVES quotes
$M_V<-9$; converting that into a stellar-mass limit needs $\Upsilon_V$, and the ELVES satellites'
own median is 1.20 against the tutorial's fiducial 1.5 — a 0.10 dex systematic shift of the
limit, landing exactly where the cumulative function is steepest.
