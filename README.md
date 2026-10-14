# LOGOS Theory A — Clinical Applications of Mathematical Recovery Models

**From Mathematical Theory to Clinical Evidence**

**Author: Vangelis D. Karalis**

**Synthetic-data release:** All packaged patient records are independently generated synthetic data. The clinical research descriptions below state the research scope; outputs from this public package are illustrative synthetic results and do not reproduce real-patient estimates or establish clinical evidence. No clinical records are read, copied, perturbed or resampled to generate these data.

**LOGOS (Longitudinal Optimization of Global Outcomes and Strategies)** is a mathematical and computational framework for evaluating whole-regimen pharmacokinetic/pharmacodynamic (PK/PD) treatment strategies.

This repository presents the clinical and computational applications of LOGOS Theory A, connecting dynamical systems, recovery-gated treatment models, statistical inference, and patient-level predictive analysis.

The study investigates how measurable biological states and treatment-withdrawal trajectories relate to subsequent clinical biomarkers and the mathematical conditions governing treatment interruption and restart.

**Research components**

- **Clinical data analysis:** Retrospective evaluation of hematologic recovery and treatment interruptions in acute lymphoblastic leukemia (ALL), alongside PSA trajectories during adaptive abiraterone therapy.
- **Mathematical and mechanistic modeling:** PK/PD simulations of drug exposure, biological responsiveness, injury repair, and disease progression.
- **Statistical inference:** Within-patient comparisons, regression modeling, sensitivity analyses, and patient-cluster bootstrap uncertainty estimation.
- **Predictive validation:** Leave-one-patient-out evaluation of whether measured restart absolute neutrophil count (ANC) improves prediction of subsequent observed hematologic minima.
- **Constrained regimen comparisons:** Computational assessment of continuous versus intermittent dosing under matched resource and safety constraints.

The accompanying Python implementations support reproducible clinical analyses, numerical modeling, comparative experiments, and verification of the reported results.

**Scientific objective:** To examine how mathematically defined recovery and restart mechanisms relate to observable clinical states, and to identify which components of model-informed treatment decisions are supported by empirical evidence.

The results provide internally validated predictive evidence for a measured hematologic safety-state component, while the clinical value of recovery-guided restart policies remains unestablished.

*Exploratory retrospective research and model-based analyses; not a validated clinical decision-support system.*

For this public release, numerical verification and internal predictive checks concern synthetic examples and specified models. Canonical parameters are normalized model scenarios; palbociclib parameters describe a typical-parameter simulation.

Version **0.5.0** supports **Python 3.12 only** (`>=3.12,<3.13`). Code, documentation and the included synthetic datasets are covered by [MIT](LICENSE). Dependency licenses remain their own.

## Reproduce in a clean environment

Run these commands from the extracted repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation .
python -m pip check
python -m unittest discover -s tests -v
logos-reproduce --data data --output results
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The workflow uses no external data downloads, author paths or spreadsheet applications. Installing dependencies requires access to a Python package index. The full run includes a 440-point phase screen and 4,000 full-training patient bootstrap draws and can take several minutes. Use a new or empty output directory for each run; failed runs retain their log.

The release includes generated input CSVs. Rebuild them or verify reproducibility with:

```bash
logos-reproduce --generate-only --data generated-data
logos-reproduce --data generated-data --output results-repeat
python scripts/compare_runs.py results results-repeat
```

The generator defaults to seed `20261012`; bootstrap stages retain their separately declared seeds. The input manifest verifies the CSV bytes before analysis. The comparison script checks every numeric CSV/JSON result, excluding execution timings and rendering caches. Rendering bytes are not portable numeric evidence.

## Included analyses

- ANC dose gaps, paired measurements, patient-cluster summaries, low-ANC no-gap matching and an adjusted observed-minimum association.
- PSA OFF episodes, landmark boundary sensitivities and strict adjacent ON-slope comparisons, including episode and patient weighting.
- Leave-one-patient-out OLS prediction with and without restart state, and 2,000 full-training patient bootstrap draws for each of two ascertainment samples. Every copy of a held-out patient is excluded from training.
- Exploratory observation-count logistic models with training-fold scaling and a fixed ridge diagnostic; visit-count and model-specification sensitivities.
- Canonical periodic and common-start finite dynamics, resource matching, mechanism controls, amplitude and parameter sweeps, phase maps and conservative whole-constant-family numerical bounds.
- Reduced waiting models with ideal and noisy recovery-rate information.
- Palbociclib oral PK/neutropenia simulation with equal administered mass and 14 days of drug-free follow-up.

Generated outputs include tables, trajectories, prediction rows, bootstrap draws, figures (PNG/SVG/PDF), `execution.log`, `run_log.json` and `verification.json`. See [model equations](docs/MODEL_SPEC.md), [data dictionary](data/README.md), [validation scope](docs/VALIDATION.md), [changelog](CHANGELOG.md) and [release instructions](RELEASE.md).

Clinical example endpoints are measured minima at simulated visits, never reconstructed continuous nadirs. Predictive improvement is not a causal restart-policy benefit. Mechanistic models and empirical example endpoints remain separate; searches do not establish global optimality. The phase map is a screened family, not a continuous safety certificate.

The GitHub About description is saved in [github-about.txt](github-about.txt), and topics are listed in [github-topics.txt](github-topics.txt). Set both in the repository's About settings after upload. A ZIP upload does not create a Git tag; use the release instructions to publish `v0.5.0`.
