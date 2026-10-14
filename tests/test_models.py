import unittest
import tempfile
from pathlib import Path
import numpy as np
from logos_apps.models import coeff, P, reference, periodic, screen
from logos_apps.synthetic import generate


class ModelTests(unittest.TestCase):
    def test_constant_stationary_equations(self):
        E, sr, dr, sz, dz=coeff(np.array([1.]),P)
        r=sr/dr;z=sz/dz
        self.assertAlmostEqual(float(sr[0]-dr[0]*r[0]),0.)
        self.assertAlmostEqual(float(sz[0]-dz[0]*z[0]),0.)
        self.assertAlmostEqual(reference()[1],2/7)

    def test_periodic_screen_converges_to_ode(self):
        ode,_=periodic(1.679033,.2)
        coarse,_=screen(1.679033,.2,N=256)
        fine,_=screen(1.679033,.2,N=2048)
        self.assertLess(abs(fine[0]-ode['J']),abs(coarse[0]-ode['J']))
        self.assertLess(abs(fine[0]-ode['J']),1e-7)
        self.assertLess(ode['closure_residual'],1e-8)

    def test_synthetic_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            generate(a);generate(b)
            for name in ['synthetic_all.csv','synthetic_psa.csv','synthetic_manifest.json']:
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())


if __name__=='__main__':unittest.main()
