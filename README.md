# orexin-swr-consolidation

Rate-based computational model linking orexinergic tone, sharp-wave ripple (SWR)
statistics, and hippocampal memory consolidation.

The model asks why pharmacological manipulations of orexin signalling produce
inconsistent behavioural effects. It separates an **encoding** module, whose gain
is scaled by global orexinergic tone, from a **consolidation** module, in which
OX2R-specific effects on SWR statistics shape replay and event-level plasticity.
Consolidation-dominant learning yields a non-monotonic (band-pass) performance
curve with an interior optimum whose location shifts with task complexity.

Code accompanying: A Computational Model Captures Orexin-Dependent Modulation of Memory Consolidation, Shakleina and Trombetta, 2026. 

Note: the code has been largely co-written by AI models and manually tested on Linux systems (Ubuntu on Google Colab, NixOS 25.05) and using Python 3.11; while no OS-specific feature is used, we recommend running this code on similar platforms.

<!--
---

## Repository layout

```
orexin-swr-consolidation/
├── orexin_paths.py            # paths + artefact identity, single source of truth
├── orexin_keys.py             # canonical library key formatter (ox_key / library_key)
├── requirements.txt
├── pyproject.toml             # uv-managed metadata (deps mirror requirements.txt)
├── common/                    # shared helpers used across the pipeline
│   ├── paths.py               #   setup(): DIRS/FILES/PROJECT_ROOT/PIPELINE_VERSION/load_artifact
│   ├── utils.py               #   ox_key, bandpass, seeding, pooling, NaN-safe summarisers
│   ├── stats.py               #   curvature/peak/bootstrap statistics (Stage 3 → 4)
│   └── plotting.py            #   plabel / panel_label / save_and_show
├── 00_datasets.py             # CRCNS hc-11 download + extract -> data_raw/
├── scripts/
│   ├── 00_migrate.py          # one-time rename of legacy versioned pickles
│   ├── 01_swr_detection.py    # LFP     -> data_processed/swr_detection.pkl
│   ├── 02_synthetic_libraries.py  # events -> models/synthetic_libraries.pkl
│   ├── 03_agent_model.py      # libraries -> results/agent_model_results.pkl
│   └── figures/               # one script per figure, each standalone
│       ├── figure_1_swr_detection.py
│       ├── figure_2_library_validation.py
│       ├── figure_3_nonmonotonic.py
│       ├── figure_4_limb_mechanisms.py
│       ├── figure_5_task_complexity.py
│       ├── figure_s1_replay_truncation.py
│       ├── figure_s2_learning_rule.py
│       └── figure_s3_seed_spread.py
├── analysis/
│   ├── detector_validation.py      # example events, distribution checks
│   ├── library_validation.py       # synthetic against empirical
│   ├── table1_calibration.py       # Table 1
│   ├── dissociation_interaction.py # formal encoding/consolidation test
│   ├── learning_rule_control.py    # Supplementary Figure 2 control
│   └── check_learning_rule_control.py
└── tests/
    └── test_model.py
```

Every script runs standalone: each reloads the artefacts it needs and assumes no
state from any other. The figure scripts each carry their own styling helpers,
so a reviewer can run one figure without running the rest. Where a helper is
genuinely shared — the key formatter, the band-pass filter, the seeding rule,
trial pooling — it lives in `orexin_keys.py` or `common/` and is imported rather
than copied.

Generated artefacts (`data_raw/`, `data_processed/`, `models/`, `results/`,
`figures/`, `tables/`, `checkpoints/`) are git-ignored and recreated by running
the scripts in order.

The scripts are the code of record. The original notebook is deliberately not
included: keeping a parallel copy invites the two to drift, and a reviewer who
ran the notebook instead of the scripts could get different numbers without
either of us noticing.

-->

## Installation

```bash
git clone https://github.com/<user>/orexin-swr-consolidation.git
cd orexin-swr-consolidation
uv sync            # creates .venv + installs the deps in pyproject.toml
```

The deps repeat `requirements.txt`; `pip install -r requirements.txt` still works if you prefer not to use uv.

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
!python scripts/figures/figure_3_nonmonotonic.py
```
</details>

## Data availability

Local field potentials come from the CRCNS `hc-11` dataset
(Grosmark, Long & Buzsáki 2016), NREM/Intermediate epochs of all POST sessions, along with their metadata (Achilles, Buddy, Cicero, Gatsby).

**These recordings are not redistributed here.** They must be downloaded from
[crcns.org](https://crcns.org) under the CRCNS data-use terms and placed in
`data_raw/<session_name>/`.


Before you start, you **must create an account on CRCNS**, and write the credentials on the
protected file `crcns-account.txt`. Then, **launch the script `00_datasets.py`** that will
automatically download and extract the required files. Bear in mind **this operation will download around 24GBs and
expand up to 150GBs**. If running on Linux systems we suggest mounting an external drive onto
`data_raw`. If running on Google Colab your only external storage would be your Google Drive.
Please make sure to run the script **from the project root folder**.

<!--
Derived artefacts (detector output, synthetic libraries, confirmed results) are
archived at `[Zenodo DOI]` for reviewers who want to reproduce the figures
without re-running the detection stage.
-->

## Reproducing the results

Check what is present, then run in order:

```bash
python orexin_paths.py          # root in use + artefact versions
```


```bash
python scripts/00_datasets.py            # Did you read the paragraph above?
python scripts/01_swr_detection.py   # requires data_raw/
python scripts/02_synthetic_libraries.py
python scripts/03_agent_model.py

python scripts/figures/figure_1_swr_detection.py
python scripts/figures/figure_2_library_validation.py
python scripts/figures/figure_3_nonmonotonic.py
python scripts/figures/figure_4_limb_mechanisms.py
python scripts/figures/figure_5_task_complexity.py
python scripts/figures/figure_s1_replay_truncation.py
python scripts/figures/figure_s3_seed_spread.py

python analysis/table1_calibration.py
python analysis/dissociation_interaction.py
```

Optional checks: `analysis/detector_validation.py` after stage 1,
`analysis/library_validation.py` after stage 2. Supplementary Figure 2 needs the
control run first:

```bash
python analysis/learning_rule_control.py --validate
python scripts/figures/figure_s2_learning_rule.py
```

Stages 2–4 are seeded (`seed=1234` for calibration, common random numbers across
OX2R levels) and reproduce the reported values exactly on the same NumPy version.

To reproduce only the figures from the archived results, download the Zenodo
artefacts into `models/` and `results/` and run the figure scripts alone.

## Model versions

Artefact filenames carry no version number. The version is written *inside*
each pickle and verified on load by `orexin_paths.load_artifact()`, which raises
on a mismatch. There is one version for the whole pipeline, declared in exactly
one place — `PIPELINE_VERSION` in `orexin_paths.py`:

| Artefact | Written by | Read by | Version field |
|---|---|---|---|
| `data_processed/swr_detection.pkl` | stage 1 | stage 2, Figure 1 | `pipeline_version` |
| `models/synthetic_libraries.pkl` | stage 2 | stage 3, Table 1 | `pipeline_version` |
| `models/generator_config.pkl` | stage 2 | stage 3 | `pipeline_version` |
| `models/validation_null.pkl` | stage 2 | Figure 2 | `pipeline_version` |
| `results/agent_model_results.pkl` | stage 3 | Figures 3–5, S3, dissociation test | `pipeline_version` |

Bumping a version means editing `PIPELINE_VERSION` and regenerating the affected
artefacts. Any script that then loads a stale pickle fails immediately rather
than silently producing a figure from the wrong data.

If you have an existing project directory with the old versioned filenames, run
`python scripts/00_migrate.py --dry-run` first, then without the flag.

## Citation

On GitHub, you can export the citation from the sidebar on the right in APA style or BibTex.
See `CITATION.cff` for details.

## License

Code released under the MIT License (see `LICENSE`).

The downloader script relies on `crcns-downloader` (hereby distributed for convenience) by Jeff Teeters, which is
also released under MIT license. CRCNS recordings remain subject
to their own terms and are not covered by this license.
 
Copyrights for Google, MATLAB, Python etc. belong to their respective companies or organizations.

