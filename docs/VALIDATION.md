# Validation scope

Validation checks specified-model computations and fictional event analyses. Passing checks supports implementation consistency, not biological adequacy or a clinical restart policy.

The fresh-run verifier reconstructs observed ANC minima and visit counts directly from event rows, verifies strict PSA selection keys, recomputes all held-out predictions with a separate centered least-squares implementation, and checks every full-training bootstrap loss using weighted design-matrix SVD. Numerical model checks compare DOP853, Radau and RK45, verify periodic closure and administration/exposure mass balance, and retain full follow-up minima.

Every bootstrap sample resamples patients, retains their episodes and excludes all copies of a held-out patient before fitting. Patient-weighted metrics average episode losses within each patient and then average patients. Episode-weighted and patient-weighted estimands are reported separately. Fixed scaling in full-training OLS is an invertible basis change, not feature selection; logistic observation models learn scaling from each training fold.

Full-training bootstrap rank failures are counted. Undefined or rank-deficient associative sensitivity estimates are retained with their reason. The synthetic observation process does not estimate or correct informative clinical sampling. Model searches retain infeasible sampled cases and negative gains.

The unit suite checks stationary equations, convergence of the calendar screen to the independently integrated periodic solution, deterministic synthetic generation and release/version consistency. CI installs the locked package in Python 3.12, performs a full run, regenerates inputs and repeats it, then compares all numeric results. The workflow is provided for GitHub; its remote execution is not asserted before a repository push.

Only Python 3.12 is supported. Dependencies are fully pinned in requirements-lock.txt. Local validation details and numerical reproduction results are recorded in REPRODUCIBILITY.md. Other operating systems are covered by the portable code and CI configuration, but are not claimed to have been locally tested.
