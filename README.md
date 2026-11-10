# Same Metric, Different Models: Interpreting Political Distances

This repository contains reusable codes and Jupyter notebooks for the manuscript's exact example and 114th U.S. Senate illustration. Start with [Notebook 1](notebooks/1_reproduce_manuscript.ipynb) to verify the saved results and regenerate Figures 1–2 and Table S1, along with an extended comparison plot. No data download or LaTeX installation is needed. Each notebook is shipped with saved figures and tables visible on opening, and rerunning its cells will create those outputs again.

## Setup

Use Python 3.11 or 3.12. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

## Notebooks

| Notebook | Purpose |
|---|---|
| [1 — Reproduce the manuscript](notebooks/1_reproduce_manuscript.ipynb) | Check the archived fits and regenerate the figures and score table. |
| [2 — Model and data](notebooks/2_model_and_data.ipynb) | Explain the models, rebuild the frozen data split, and verify the exact construction. |
| [3 — Refit the Senate](notebooks/3_refit_senate.ipynb) | Estimate both models at all three coordinate-prior scales and compare with the archived results. |

Notebook 3 defaults to `MODE = "full"` and includes saved outputs from the complete fitting procedure. For a short execution check, set `MODE = "smoke"` in its configuration cell; this uses fewer starts and a 30-iteration cap. Smoke estimates are not the manuscript results. A full run took approximately 82 seconds in the verification environment; runtime varies by machine.

Each `Run All` creates a new output directory. To repeat the fitting cell alone, first rerun the configuration cell. New results go into `outputs/`, which is created automatically; the bundled data and reference fits are preserved.

## Contents

```text
notebooks/        Three executable notebooks
idealpoint/       Shared fitting, checking, and plotting code
data/             Voteview snapshot, processed data, and fixed split
reference/        Saved fits, diagnostics, original figures, and provenance
requirements.txt  Pinned dependencies
```

The notebooks include the verification steps. Optional commands from this folder also reproduce the displays or rerun estimation:

```bash
python -m idealpoint.figures --source reference --output outputs/reference-figures
python -m idealpoint.study --mode full --output outputs/full-refit
```

Use a new output name for each command-line refit. Original manuscript exports are retained in `reference/manuscript_figures/`.

## Data and analysis

The [Voteview](https://voteview.com/data) snapshot covers the 114th Senate (2015-2016), retrieved October 23, 2026 UTC. After filtering, 96 senators and 373 roll calls contribute 28,164 training votes and 7,018 held-out votes. Only Yea/Nay codes 1 and 6 enter; the outcome-balance filter uses training votes only. Source URLs, hashes, filters, and seeds are recorded in `data/fit_comparison_design.json`.

Both models use `logit(p_ij) = a_i + b_j - ||z_i - w_j||^q`, with `q` equal to 1 or 2, two latent dimensions, intercept prior SD 3, and coordinate prior SD in `{0.5, 1, 2}`. Starts are selected by the penalized training objective. Party labels are used after fitting for display and descriptive comparisons.

These are approximate joint MAP fits on one fixed split. The same parameter priors induce different probability priors. The archived parameters and logits matched a complete rerun exactly in the verification environment (Python 3.12.13 on macOS, November 7, 2026); other platforms can produce numerical differences. Provenance is recorded in `reference/provenance.json`.

Data citation: Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet. 2026. *Voteview: Congressional Roll-Call Votes Database*. https://voteview.com/.