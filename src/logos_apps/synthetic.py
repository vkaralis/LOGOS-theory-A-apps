"""Generate independent fictional records; no source records are read or resampled."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SEED = 20261012


def generate(destination: Path, seed: int = SEED) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, 61):
        bsa = float(rng.uniform(.65, 1.6))
        weight = float(rng.uniform(18, 60))
        height = float(rng.uniform(105, 170))
        dose = float(rng.uniform(25, 70) * bsa)
        gaps = [(start, start + int(rng.integers(4, 10)))
                for start in [30, 85, 140, 195]]
        days = [day for day in range(241)
                if not any(start <= day < end for start, end in gaps)]
        for day in days:
            rows.append(dict(ID=pid, TIME=day, EVID=1, CMT=1, MDV=1,
                             AMT=float(dose*rng.uniform(.85,1.15)), DV=0., BSA=bsa, WEIGHT=weight, HEIGHT=height))
        visits = set(range(0, 241, 7))
        for start, end in gaps:
            visits.update([start - 1, end - 1])
            # Include missing, one-visit and multiple-visit follow-up windows.
            visits.difference_update(range(end + 1, end + 15))
            n = int(rng.choice([0, 1, 2, 3], p=[.08, .17, .55, .20]))
            visits.update(int(end + d) for d in [3, 8, 13][:n])
        host = float(rng.uniform(.8, 1.8))
        for day in sorted(visits):
            anc = host + .65 * np.sin(day / 18 + pid) + rng.normal(0, .22)
            for start, end in gaps:
                if day == start - 1:
                    anc = float(rng.uniform(.1, .45) if pid % 3 == 0 else rng.uniform(.5, 2.0))
                elif day == end - 1:
                    anc += .4 + .06 * (end - start)
            rows.append(dict(ID=pid, TIME=day, EVID=0, CMT=8, MDV=0,
                             AMT=0., DV=max(.05, float(anc)), BSA=bsa,
                             WEIGHT=weight, HEIGHT=height))
    pd.DataFrame(rows).sort_values(['ID', 'TIME', 'EVID']).to_csv(
        destination / 'synthetic_all.csv', index=False, float_format='%.12g')
    rows = []
    for pid in range(40):
        cohort = 'Adaptive' if pid < 30 else 'SOC'
        reference = float(rng.uniform(8, 40))
        logpsa = float(np.log(reference))
        time = 0.
        phases = [1, 0, 1, 0, 1, 0, 1] if cohort == 'Adaptive' else [1]
        for phase_index, on in enumerate(phases):
            duration = float(rng.integers(65, 151) if on else rng.integers(40, 111))
            # A few sparse ON phases exercise strict slope exclusions.
            samples = 2 if pid % 7 == 0 and on else int(rng.integers(4, 9))
            rate = float(rng.uniform(-.018, -.004) if on else rng.uniform(.004, .02))
            offsets = np.linspace(0, duration, samples + 1)[:-1]
            for offset in offsets:
                value = float(np.exp(logpsa + rate * offset + rng.normal(0, .07)))
                rows.append(dict(patient=f'SYN_PSA_{pid+1:03d}', cohort=cohort,
                                 time=time + offset, PSA=value, on=on,
                                 relative_PSA=value/reference))
            logpsa += rate * duration
            time += duration
        value = float(np.exp(logpsa))
        rows.append(dict(patient=f'SYN_PSA_{pid+1:03d}', cohort=cohort,
                         time=time, PSA=value, on=phases[-1], relative_PSA=value/reference))
    pd.DataFrame(rows).sort_values(['patient', 'time']).to_csv(
        destination / 'synthetic_psa.csv', index=False, float_format='%.12g')
    manifest = dict(synthetic=True, seed=seed,
                    method='Independent fictional event histories; no clinical input files used',
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(destination.glob('synthetic_*.csv'))})
    (destination / 'synthetic_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
