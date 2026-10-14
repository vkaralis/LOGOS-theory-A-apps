"""Independent weighted SVD audit of every full-training bootstrap draw."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def audit(out: Path):
    panel=pd.read_csv(out/'ALL_restart_panel.csv')
    panel['gap_days']=panel.restart_time-panel.gap_start
    common=['pre_gap_ANC','restart_dose_mgm2','gap_days','restart_time']
    extra=common+['pre_restart_ANC']
    errors={}
    for label,z in [('all',panel),('two_or_more_visits',panel[panel.n_visits>=2])]:
        z=z.dropna(subset=extra+['minimum_ANC'])
        x=z[extra].to_numpy(float)
        x=np.column_stack([np.ones(len(z)),(x-x.mean(0))/x.std(0)])
        y=z.minimum_ANC.to_numpy(float);ids=z.patient.to_numpy()
        patients=np.sort(np.unique(ids));indices=[np.flatnonzero(ids==p) for p in patients]
        rng=np.random.default_rng(20261010)
        reference=pd.read_csv(out/f'A_training_bootstrap_{label}.csv').set_index('replicate')
        maxerror=0.;valid=0;invalid=0
        for rep in range(2000):
            counts=np.bincount(rng.integers(len(patients),size=len(patients)),minlength=len(patients))
            losses=np.zeros(2);good=True
            weights=np.zeros(len(z))
            for i,ix in enumerate(indices):weights[ix]=counts[i]
            for i,ix in enumerate(indices):
                if counts[i]==0:continue
                # Exclude ALL copies of the patient; bootstrap replication is a weight.
                train=weights.copy();train[ix]=0
                selected=train>0
                w=np.sqrt(train[selected])
                for j,ncols in enumerate([len(common)+1,len(extra)+1]):
                    a=x[selected,:ncols]*w[:,None];b=y[selected]*w
                    beta,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
                    if rank<ncols:good=False;break
                    pred=x[ix,:ncols]@beta
                    losses[j]+=counts[i]*float(np.mean((y[ix]-pred)**2))/len(patients)
                if not good:break
            if good:
                valid+=1
                expected=reference.loc[rep,['MSE_baseline','MSE_state','MSE_reduction']].to_numpy(float)
                maxerror=max(maxerror,float(abs(np.r_[losses,losses[0]-losses[1]]-expected).max()))
            else:
                invalid+=1
                if rep in reference.index:raise AssertionError('Rank-failure disagreement')
        if valid!=len(reference):raise AssertionError('Bootstrap draw-count disagreement')
        errors[label]=dict(valid=valid,invalid=invalid,max_absolute_loss_error=maxerror,
                           passed=bool(maxerror<1e-8))
    result=dict(method='Weighted design matrix SVD, independently excluding all copies of test patient',
                all_passed=all(v['passed'] for v in errors.values()),samples=errors)
    (out/'bootstrap_independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_passed']:raise AssertionError(result)
    return result
