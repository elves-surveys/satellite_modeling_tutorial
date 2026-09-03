# From Subhalos to Observed Satellites

*Jiaxuan Li ([jiaxuanl@stanford.edu](mailto:jiaxuanl@stanford.edu)) —
[github.com/elves-surveys/satellite_modeling_tutorial](https://github.com/elves-surveys/satellite_modeling_tutorial)*

A hands-on tutorial on forward-modeling the satellite galaxy populations of Milky-Way-mass
hosts — from a dark-matter subhalo catalog, through galaxy occupation and the stellar-to-halo
mass relation, through observable properties and viewing geometry, to a mock that can be
compared with a real survey.

Built for the KICP workshop
["Probe Combination for Dark Matter Physics in the Era of Large Surveys"](https://indico.uchicago.edu/event/580/overview).

> **This is a toy model demo, not a comprehensive modeling attempt.** Every relation used here
> — the occupation fraction, the stellar-to-halo mass relation, galaxy size, mass-to-light —
> is a simple functional form with parameters fixed by hand for transparency and speed, not
> fit to be a state-of-the-art prediction. The comparisons to ELVES and the Milky Way census in
> Parts 3–4 are illustrations of a method, not results: every discrepancy they turn up comes
> with a list of simplifications (no disk, no bulge, a hand-picked SHMR, ...) that could easily
> explain it before any dark-matter physics does. Don't take the numbers that come out of these
> notebooks at face value — the point is the forward-modeling chain, not the specific answer.

## Structure

```
notebooks/   the tutorial itself, run in order
src/         the forward-model code the notebooks call (see src/README.md)
data/        SatGen subhalo/host catalogs (parquet)
data_obs/    small observational tables (MW, ELVES, DELVE MW census)
```

| notebook | part | covers |
|---|---|---|
| [01_subhalos.ipynb](notebooks/01_subhalos.ipynb) | Part 1 | the subhalo catalog, galaxy occupation, the SHMR |
| [02_satellites.ipynb](notebooks/02_satellites.ipynb) | Part 2 | observable properties ($M_V$, size, surface brightness) and projection/viewing geometry |
| [03_elves_comparison.ipynb](notebooks/03_elves_comparison.ipynb) | Part 3 | comparing the mock to ELVES, following Danieli et al. (2023) |
| [04_mw_sats_comparison.ipynb](notebooks/04_mw_sats_comparison.ipynb) | Part 4 (optional) | forward-modeling the Milky Way itself against the published DES/DELVE selection function (DELVE MW census) |

Each notebook is self-contained (re-imports and reloads data at the top) and ends with a
"Try it" / Exercises section — read through once, then go back and turn the knobs.

## Running it

### Option A: GitHub Codespaces (recommended for the workshop)

1. On the repo's GitHub page: **Code → Codespaces → Create codespace on main**.
2. Wait for the container to build and `pip install -r requirements.txt` to finish
   (a minute or two) — this happens automatically via `.devcontainer/devcontainer.json`.
3. Open a notebook in `notebooks/` and select the default Python kernel when prompted.

No local Python install, no dependency wrangling — everything (including the data) is already
in the repo.

### Option B: run locally

Requires Python ≥3.10.

```bash
git clone https://github.com/elves-surveys/satellite_modeling_tutorial.git
cd satellite_modeling_tutorial
python -m venv .venv && source .venv/bin/activate      # or conda/mamba
pip install -r requirements.txt
jupyter lab notebooks/
```

## Data

`data/` holds two SatGen parquet tables (`satgen_new_MW_hosts.parquet`,
`satgen_new_MW_subhalos.parquet`), used by every notebook; `data_obs/` holds small
observational tables — the ELVES host/satellite catalogs (Part 3) and the DELVE Milky Way
census release (injection-recovery sims, footprint, satellites, posterior chain — Part 4). Both
directories are small enough to be committed directly — no external download step needed. See
`src/README.md` for what each file contains and the `KICP_DATA_DIR` / `KICP_OBS_DIR`
environment variables if you want to point the notebooks at a different copy.

## License

MIT — see [LICENSE](LICENSE).
