"""Compare all numeric CSV/JSON outputs; timings and image bytes are excluded."""
import argparse
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
a=ap.parse_args()
def inventory(root):
    return {p.relative_to(root) for p in root.rglob('*') if p.suffix in ['.csv','.json']
            and '.matplotlib' not in p.parts and p.name!='run_log.json'}
first,second=inventory(a.first),inventory(a.second)
assert first==second,first.symmetric_difference(second)
for rel in sorted(first):
    x,y=a.first/rel,a.second/rel
    if x.suffix=='.json':same(json.loads(x.read_text()),json.loads(y.read_text()),str(rel))
    else:
        # Match empty result tables as well as populated tables.
        if x.stat().st_size<2 or y.stat().st_size<2:
            assert x.read_bytes()==y.read_bytes(),rel
        else:
            pd.testing.assert_frame_equal(pd.read_csv(x),pd.read_csv(y),check_exact=False,
                                          rtol=1e-9,atol=1e-10)
print(f'{len(first)} numeric result files agree')
