"""Build a deterministic source ZIP without private files or Git metadata."""
import argparse
import hashlib
from pathlib import Path
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

root=Path(__file__).resolve().parents[1]
version=tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=root.parent/f'logos-theory-a-apps-v{version}.zip')
args=ap.parse_args();target=args.output.resolve();target.parent.mkdir(parents=True,exist_ok=True)
allowed=['.github','data','docs','examples','scripts','src','tests']
files=[p for p in root.iterdir() if p.is_file() and p.suffix not in ['.zip','.bundle']]
for name in allowed:
    files += [p for p in (root/name).rglob('*') if p.is_file() and
              not any(x in ['__pycache__','build','dist'] or x.endswith('.egg-info') for x in p.parts)
              and p.name!='.DS_Store' and p.suffix not in ['.pyc','.pyo']]
with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=9) as archive:
    for p in sorted(files):
        info=ZipInfo(f'logos-theory-a-apps/{p.relative_to(root).as_posix()}',date_time=(1980,1,1,0,0,0))
        info.external_attr=0o100644<<16;info.compress_type=ZIP_DEFLATED
        archive.writestr(info,p.read_bytes())
sha=hashlib.sha256(target.read_bytes()).hexdigest()
target.with_suffix('.zip.sha256').write_text(f'{sha}  {target.name}\n')
print(f'{target}\nSHA-256 {sha}\n{len(files)} files')
