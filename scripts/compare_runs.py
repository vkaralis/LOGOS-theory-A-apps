"""Compare all numeric CSV/JSON outputs; timings and image bytes are excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def same(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for key in a:same(a[key],b[key],path+'/'+key)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'/{i}')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-9,atol=1e-10),f'{path}: {a} != {b}'
    else:assert a==b,path


ap=argparse.ArgumentParser();ap.add_argument('first',type=Path);ap.add_argument('second',type=Path)
ap.add_argument('--first-data',type=Path)
ap.add_argument('--second-data',type=Path)
a=ap.parse_args()
input_hashes=None
if (a.first_data is None)!=(a.second_data is None):
    ap.error('Provide both --first-data and --second-data')
if a.first_data is not None:
    manifests=[]
    for root in [a.first_data,a.second_data]:
        manifest=json.loads((root/'synthetic_manifest.json').read_text())
        assert manifest['synthetic'] is True,'Inputs must be synthetic'
        assert set(manifest['files'])=={'synthetic_all.csv','synthetic_psa.csv'},'Input inventory mismatch'
        for name,digest in manifest['files'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,f'Input hash mismatch: {root/name}'
        manifests.append(manifest)
    same({k:v for k,v in manifests[0].items() if k!='files'},
         {k:v for k,v in manifests[1].items() if k!='files'},'input manifests')
    for name in sorted(manifests[0]['files']):
        pd.testing.assert_frame_equal(pd.read_csv(a.first_data/name),pd.read_csv(a.second_data/name),
                                      check_exact=False,rtol=1e-9,atol=1e-10)
    input_hashes=[m['files'] for m in manifests]
    print('Both input manifests verified; synthetic CSVs numerically agree')
def inventory(root):
    return {p.relative_to(root) for p in root.rglob('*') if p.suffix in ['.csv','.json']
            and '.matplotlib' not in p.parts and p.name!='run_log.json'}
first,second=inventory(a.first),inventory(a.second)
assert first==second,first.symmetric_difference(second)
for rel in sorted(first):
    x,y=a.first/rel,a.second/rel
    if x.suffix=='.json':
        left,right=json.loads(x.read_text()),json.loads(y.read_text())
        if rel==Path('empirical_summary.json') and input_hashes is not None:
            # Floating-point CSV serialization can differ across operating systems.
            # Accept differing hashes only after verifying each manifest, comparing
            # every input value, and checking the provenance against its own inputs.
            for record,expected in zip([left,right],input_hashes):
                assert record['source_SHA256']==expected,'Result/input provenance mismatch'
            left=dict(left);right=dict(right)
            left.pop('source_SHA256');right.pop('source_SHA256')
        same(left,right,str(rel))
    else:
        # Match empty result tables as well as populated tables.
        if x.stat().st_size<2 or y.stat().st_size<2:
            assert x.read_bytes()==y.read_bytes(),rel
        else:
            pd.testing.assert_frame_equal(pd.read_csv(x),pd.read_csv(y),check_exact=False,
                                          rtol=1e-9,atol=1e-10)
print(f'{len(first)} numeric result files agree')
