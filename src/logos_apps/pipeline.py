"""Portable staged execution and provenance recording."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description='Reproduce LOGOS A synthetic examples')
    ap.add_argument('--data', type=Path, default=Path('data'))
    ap.add_argument('--output', type=Path, default=Path('results'))
    ap.add_argument('--generate-only', action='store_true')
    args = ap.parse_args()
    from .synthetic import generate
    data, out = args.data.resolve(), args.output.resolve()
    if args.generate_only:
        generate(data)
        return
    if out.exists() and any(out.iterdir()):
        ap.error('Output must be a new or empty directory to avoid stale results')
    out.mkdir(parents=True, exist_ok=True)
    if not (data / 'synthetic_manifest.json').exists():
        ap.error('Generate synthetic inputs first with --generate-only')
    manifest = json.loads((data / 'synthetic_manifest.json').read_text())
    if manifest.get('synthetic') is not True:
        ap.error('This release requires explicitly synthetic inputs')
    for name in ['synthetic_all.csv', 'synthetic_psa.csv']:
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != manifest['files'][name]:
            ap.error(f'Input hash mismatch: {name}')
    env = os.environ.copy()
    env.update(LOGOS_DATA=str(data), LOGOS_OUTPUT=str(out),
               MPLCONFIGDIR=str(out / '.matplotlib'), MPLBACKEND='Agg',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    stages = [('empirical', ['--output', str(out)]), ('clinical', []),
              ('models', ['--output', str(out)]), ('extensions', []),
              ('finite_family', []), ('training', []), ('diagnostics', []),
              ('verify', [])]
    log = dict(version='0.5.0', python=platform.python_version(),
               platform=platform.platform(), synthetic=True, inputs=manifest,
               dependencies={k: importlib.metadata.version(k)
                             for k in ['numpy', 'pandas', 'scipy', 'matplotlib']},
               stages=[])
    with (out / 'execution.log').open('w') as stream:
        for module, options in stages:
            print(f'Running {module}', flush=True)
            start = time.monotonic()
            proc = subprocess.run([sys.executable, '-m', f'logos_apps.{module}', *options],
                                  env=env, stdout=stream, stderr=subprocess.STDOUT)
            log['stages'].append(dict(module=module, returncode=proc.returncode,
                                      seconds=time.monotonic()-start))
            log['success'] = proc.returncode == 0 and len(log['stages']) == len(stages)
            (out/'run_log.json').write_text(json.dumps(log, indent=2)+'\n')
            if proc.returncode:
                raise SystemExit(f'{module} failed; see {out / "execution.log"}')
    print(f'All stages passed. Results: {out}', flush=True)


if __name__ == '__main__':
    main()
