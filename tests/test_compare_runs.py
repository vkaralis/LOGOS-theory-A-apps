"""Input integrity and cross-platform numeric reproduction checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ComparisonTests(unittest.TestCase):
    def check_comparison(self, difference=1e-12, corrupt=False, wrong_provenance=False):
        script=Path(__file__).resolve().parents[1]/'scripts/compare_runs.py'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for side,value in [('a',1.25),('b',1.25+difference)]:
                data=root/side/'data';out=root/side/'results'
                data.mkdir(parents=True);out.mkdir()
                hashes={}
                for name in ['synthetic_all.csv','synthetic_psa.csv']:
                    text=f'patient,value\nSYN_001,{value:.15g}\n'
                    (data/name).write_text(text)
                    hashes[name]=hashlib.sha256(text.encode()).hexdigest()
                manifest=dict(synthetic=True,seed=20261012,method='Fictional test data',files=hashes)
                (data/'synthetic_manifest.json').write_text(json.dumps(manifest))
                provenance=dict(hashes)
                if wrong_provenance and side=='b':provenance['synthetic_all.csv']='wrong'
                (out/'empirical_summary.json').write_text(json.dumps(dict(source_SHA256=provenance,estimate=2.)))
                if corrupt and side=='b':(data/'synthetic_all.csv').write_text('patient,value\nSYN_001,9\n')
            return subprocess.run([sys.executable,str(script),str(root/'a/results'),str(root/'b/results'),
                                   '--first-data',str(root/'a/data'),'--second-data',str(root/'b/data')],
                                  capture_output=True,text=True)

    def test_rounding_difference_with_valid_manifests_passes(self):
        result=self.check_comparison()
        self.assertEqual(result.returncode,0,result.stderr)

    def test_changed_data_fails(self):
        self.assertNotEqual(self.check_comparison(difference=.01).returncode,0)

    def test_corrupt_manifest_fails(self):
        self.assertNotEqual(self.check_comparison(corrupt=True).returncode,0)

    def test_wrong_result_provenance_fails(self):
        self.assertNotEqual(self.check_comparison(wrong_provenance=True).returncode,0)
