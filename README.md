# orexin-swr-consolidation

Rate-based computational model linking orexinergic tone, sharp-wave ripple (SWR)
statistics, and hippocampal memory consolidation.

The model asks why pharmacological manipulations of orexin signalling produce
inconsistent behavioural effects. It separates an **encoding** module, whose gain
is scaled by global orexinergic tone, from a **consolidation** module, in which
OX2R-specific effects on SWR statistics shape replay and event-level plasticity.
Consolidation-dominant learning yields a non-monotonic (band-pass) performance
curve with an interior optimum whose location shifts with task complexity.

Code accompanying: *[manuscript title]*, [authors], [year]. `[DOI or preprint link]`

---

## Repository layout

```
orexin-swr-consolidation/
├── orexin_paths.py            # paths + artefact versions, single source of truth
├── requirements.txt
├── scripts/
│   ├── 00_migrate_filenames.py     # one-time rename of legacy versioned pickles
│   ├── 01_swr_detection.py         # detector  → swr_detection.pkl
│   ├── 02_synthetic_generator.py   # generator → synthetic_libraries.pkl
│   ├── 03_agent_model.py           # agent     → agent_model_results.pkl
│   └── 04_figures.py               # Figures 1–5 + supplementary
├── analysis/
│   └── build_table1_calibration.py
└── tests/
    └── test_model.py
```

Generated artefacts (`data_processed/`, `models/`, `results/`, `figures/`,
`tables/`) are git-ignored and recreated by running the scripts in order.

The scripts are the code of record. The original notebook is deliberately not
included: keeping a parallel copy invites the two to drift, and a reviewer who
ran the notebook instead of the scripts could get different numbers without
either of us noticing.

## Installation

```bash
git clone https://github.com/<user>/orexin-swr-consolidation.git
cd orexin-swr-consolidation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.11. Plain Python throughout — no notebook runtime, no cloud
storage, no machine-specific paths.

### Where the data lives

By default the pipeline reads and writes inside the repository directory, so a
fresh clone runs with no configuration. To keep data elsewhere, set one
variable:

```bash
export OREXIN_PROJECT_ROOT=/path/to/orexin_spw_r_project
```

That is the only knob. `python orexin_paths.py` reports which root is in use
and which artefacts are present.

<details>
<summary>Running in a hosted notebook</summary>

Nothing Colab-specific ships in this repository. If you want to run it in a
hosted notebook backed by Google Drive, mount the drive yourself and point the
environment variable at it before importing anything:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['OREXIN_PROJECT_ROOT'] = '/content/drive/MyDrive/orexin_spw_r_project'

!pip install -q -r requirements.txt
!python scripts/04_figures.py
```
</details>

## Data availability

Local field potentials come from the CRCNS `hc-[XX]` dataset
(Girardeau et al., 2017), NREM/Intermediate epochs of three POST sessions:
`Achilles_10252013`, `Cicero_09102014`, `Cicero_09172014`.

**These recordings are not redistributed here.** They must be downloaded from
[crcns.org](https://crcns.org) under the CRCNS data-use terms and placed in
`data_raw/<session_name>/`. Everything downstream is reproducible from them.

Derived artefacts (detector output, synthetic libraries, confirmed results) are
archived at `[Zenodo DOI]` for reviewers who want to reproduce the figures
without re-running the detection stage.

## Reproducing the results

Check what is present, then run in order:

```bash
python orexin_paths.py          # root in use + artefact versions
```


```bash
python scripts/00_migrate_filenames.py   # once, if migrating an old directory
python scripts/01_swr_detection.py       # ~[X] min; requires data_raw/
python scripts/02_synthetic_generator.py # ~[X] min
python scripts/03_agent_model.py         # ~[X] h
python scripts/04_figures.py             # writes figures/
```

Stages 2–4 are seeded (`seed=1234` for calibration, common random numbers across
OX2R levels) and reproduce the reported values exactly on the same NumPy version.

To reproduce only the figures from the archived results, download the Zenodo
artefacts into `models/` and `results/` and run stage 04 alone.

## Model versions

Artefact filenames carry no version number. The version is written *inside*
each pickle and verified on load by `orexin_paths.load_artifact()`, which raises
on a mismatch. Versions are declared in exactly one place — `VERSIONS` in
`orexin_paths.py`:

| Component | Version | Artefact | Version field |
|---|---|---|---|
| SWR detection | 5.4.5 | `swr_detection.pkl` | `version` |
| Synthetic SPW/ripple generator | 5.8.3 | `synthetic_libraries.pkl` | `generator_version` |
| | | `generator_config.pkl` | `generator_version` |
| Agent model | 2.5.11 | `agent_model_results.pkl` | `version` |

Bumping a version means editing `VERSIONS` and regenerating the affected
artefact. Any script that then loads a stale pickle fails immediately rather
than silently producing a figure from the wrong data.

If you have an existing project directory with the old versioned filenames, run
`python scripts/00_migrate_filenames.py --dry-run` first, then without the flag.

## Citation

See `CITATION.cff`.

## License

Code released under the MIT License (`LICENSE`). CRCNS recordings remain subject
to their own terms and are not covered by this license.
