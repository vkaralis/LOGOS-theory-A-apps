# Output guide

All clinical/example outputs refer to the packaged synthetic records. Files are generated into a user-selected directory. Tables and trajectories use `tables/`; plots use `figures/`. A run ends successfully only when every stage and the verifier pass.

| Stage | Principal outputs |
| --- | --- |
| empirical | `empirical_summary.json`, `T01_dataset_inventory.csv`, `T02_ALL_gap_sensitivity.csv`, `T03_abiraterone_summary.csv`, raw-derived gap/PSA tables and descriptive plots |
| clinical | `results.json` (matched, associative and strict PSA summaries), `ALL_restart_panel.csv`, no-gap candidates, matched pairs, strict/inclusive PSA slope pairs |
| models | `model_summary.json`, calendar grid, convergence, mechanism/amplitude/robustness tables, phase map, finite trajectories, reduced waiting and palbociclib results |
| extensions | `A_constant_family.json`, `A_robustness.json`, `A_imperfect_information.json`, `A_clinical_prediction.json`, all held-out predictions and patient losses |
| finite_family | `A_finite_constant_family.json` with common-state constant-family upper bound, cycle comparison and solver/path checks |
| training | `A_training_bootstrap_all.csv`, `A_training_bootstrap_two_or_more_visits.csv`, `A_full_training_uncertainty.json` |
| diagnostics | timing, visit-count, selection, dose-specification and landmark sensitivities, included/excluded descriptors, observation-model coefficients/predictions, patient-cluster SE, delayed-tail tables and figures |
| verify | `verification.json`, `bootstrap_independent_verification.json`, `candidate_register.json`, four named periodic traces |

`run_log.json` records the environment, input manifest, return codes and stage timings; `execution.log` retains detailed stage output. Numerical CSV missing values are blank, while undefined JSON estimates use null. Optimization and rank-failure records are retained. OLS predictions are unconstrained; a negative prediction is reported rather than silently clipped.

Units: ANC is G/L; ALL doses are mg or mg/m² where declared; PSA slopes are 1/day; palbociclib concentration is µg/L and time days; canonical time, input, injury and burden are normalized. `results.json` contains synthetic clinical associations; `model_summary.json` contains specified-model outputs. Combining their endpoints into a single clinical tradeoff is not supported.
