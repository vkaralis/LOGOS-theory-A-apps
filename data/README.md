# Synthetic data

`synthetic_all.csv` contains 60 fictional patient dose and ANC event histories. `synthetic_psa.csv` contains 40 fictional PSA histories (30 Adaptive, 10 SOC). Counts are chosen for examples and do not stand for clinical cohort sizes. `synthetic_manifest.json` records the seed and SHA-256 input hashes. All files are MIT licensed.

The generator reads no real dataset. ALL uses independent anthropometrics, dose levels, four calendar gaps, noisy ANC trajectories and varying follow-up visit counts. PSA uses alternating treatment phases, random log-linear rates, noisy measurements and occasional sparse ON segments. Distributions are arbitrary teaching scenarios; missing follow-up is simulated by omitting visits. Restart-panel outcomes are derived from the resulting raw events.

| ALL column | Meaning / unit |
| --- | --- |
| ID | Fictional integer patient ID |
| TIME | Time in days |
| EVID | 1 dose event; 0 ANC observation |
| CMT | 1 administration compartment; 8 ANC observation |
| MDV | 1 dose row without an observed ANC; 0 measured row |
| AMT | Administered amount in mg; zero on observation rows |
| DV | ANC in G/L on EVID=0; zero placeholder on dose rows |
| BSA | Body surface area in m² |
| WEIGHT | Weight in kg |
| HEIGHT | Height in cm |

| PSA column | Meaning / unit |
| --- | --- |
| patient | Fictional ID with SYN_PSA prefix |
| cohort | Adaptive or SOC synthetic label |
| time | Time in days |
| PSA | Positive synthetic PSA value, arbitrary common units |
| on | Binary recorded administration status |
| relative_PSA | PSA divided by that fictional patient's generating reference |

Do not interpret dose rows' DV=0 as zero ANC. Dose gaps are inferred from consecutive records separated by more than 1.5 days. A pre-gap value must lie within seven days before gap start; a pre-restart value must lie within the gap and within seven days before restart. Paired measurements must have strictly increasing times. The prediction outcome is the observed minimum strictly after restart through day 14. No visits means missing outcome, not zero.

PSA slope estimates need positive measurements and distinct times. Strict ON segments exclude the next OFF switch measurement. The primary landmark chooses the closest recorded observation to restart+60 in days 28–90; a separate sensitivity excludes the next OFF boundary. Treatment status between observations uses a left-continuous convention.
