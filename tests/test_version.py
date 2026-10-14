import os
from pathlib import Path
import tomllib
import unittest
import logos_apps


class VersionTests(unittest.TestCase):
    def test_version_and_python_range(self):
        root=Path(__file__).resolve().parents[1]
        meta=tomllib.loads((root/'pyproject.toml').read_text())['project']
        self.assertEqual(meta['version'],logos_apps.__version__)
        self.assertEqual(meta['requires-python'],'>=3.12,<3.13')
        self.assertIn('[0.5.0]',(root/'CHANGELOG.md').read_text())
        if os.environ.get('GITHUB_REF_TYPE')=='tag':
            self.assertEqual(os.environ['GITHUB_REF_NAME'],'v'+meta['version'])
