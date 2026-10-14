# Executed reproducibility record

Validated on 12 October 2026 with a newly created virtual environment, Python 3.12.14 and macOS arm64. The environment was created without system-site-packages. Dependencies were installed from the package index; the project was built and installed as a wheel. `pip check` reported no broken requirements.

The final installed package completed all eight stages successfully: empirical events, clinical contrasts, model campaign, model/prediction extensions, finite constant-family benchmark, full-training bootstrap, diagnostics and independent verification. The first final campaign took approximately 157 seconds on the validation host. Runtime varies by machine.

- Four unit/version tests passed.
- All 315 fresh-run checks passed.
- Every held-out prediction was independently recomputed.
- All 4,000 full-training bootstrap draws were checked with a separate weighted design-matrix SVD implementation. Maximum absolute loss differences were 1.20×10⁻¹⁵ and 1.00×10⁻¹⁵ for the two samples, with no rank failures.
- Synthetic inputs were regenerated in a separate directory and a second complete campaign passed. All 79 numeric CSV/JSON result files agreed within rtol=1e−9, atol=1e−10.
- Selected generated clinical and mechanistic figures were visually inspected. Image/PDF bytes are not included in numeric agreement claims.

`examples/expected_summary.json` retains model and synthetic prediction summaries; `examples/verification.json` retains the final check details. These examples describe the packaged synthetic data only. The Python 3.12 GitHub Actions workflow is configured but has not been executed on a remote repository during local preparation. No other Python version or operating system is claimed as locally tested.

On 14 October 2026, the initial release ZIP was independently installed as a wheel in a new isolated Python 3.12.14 environment on macOS arm64 with the locked dependencies. Both complete runs passed all 315 checks, and all 79 numeric result files agreed. The four unit/version tests passed. Subsequent author, README and packaging updates preserved the model equations and input data. The unit/version suite passed again, and generated PNG, SVG and PDF files were checked for absence of embedded software/creator/date metadata. Those presentation updates did not trigger a new complete numerical campaign.

Version 0.5.1 corrects the cross-platform input comparison. The initial Ubuntu run completed both complete analysis campaigns, but its comparison failed because regenerated CSV hashes differed from the macOS-generated inputs. Each manifest is now verified against its own CSV bytes, every input value is compared numerically, and each result hash must match its own verified input manifest. Model equations and packaged datasets are unchanged. Four comparison regression tests supplement the original four unit/version tests.
