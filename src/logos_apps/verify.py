"""Independent numeric and selection checks for a fresh synthetic run."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .models import periodic, finite, reference

OUT=Path(os.environ['LOGOS_OUTPUT'])
checks=[]


def check(name, error, tolerance):
    checks.append(dict(check=name,error=float(error),tolerance=tolerance,passed=bool(error<=tolerance)))


def main():
    v,_=periodic(1.679033,.2)
    alt,_=periodic(1.679033,.2,method='Radau',rtol=2e-12)
    check('periodic alternative solver objective',abs(v['J']-alt['J']),1e-8)
    check('periodic alternative solver injury',abs(v['peak_Z']-alt['peak_Z']),1e-8)
    check('periodic closure',v['closure_residual'],1e-8)
    check('periodic objective numerical anchor',abs(v['J']-.080905389814),1e-9)
    check('constant objective numerical anchor',abs(reference()[0]-(-.032715501)),1e-9)
    register=[]
    for name,period in [('A-PER-REFINED',1.679033),('A-PER-RERUN',1.679027411422),('A-AMP-5',1.679036617960),('A-CAP-EXT',1.679903)]:
        candidate,trace=periodic(period,.2)
        candidate['name']=name
        candidate['initial_state']=trace.iloc[0][['C','R','Z']].to_dict()
        trace.to_csv(OUT/'tables'/f'{name}_trace.csv',index=False)
        register.append(candidate)
    (OUT/'candidate_register.json').write_text(json.dumps(register,indent=2)+'\n')
    a,_=finite();b,_=finite(method='Radau',rtol=2e-12)
    check('finite alternative solver objective',float(abs(a.J_F-b.J_F).max()),1e-8)
    check('finite exposure mass balance',float(a.mass_balance_error.max()),1e-8)
    pal=pd.read_csv(OUT/'tables/palbociclib_tail.csv')
    alt=pd.read_csv(OUT/'tables/palbociclib_tail_independent.csv')
    check('oral PK mass balance',float(pal.PK_mass_balance_error.max()),1e-7)
    check('palbociclib alternative solver nadir',float(abs(pal.nadir_0_182-alt.nadir_0_182).max()),1e-7)
    check('equal administered palbociclib amounts',float(abs(pal.administered_mg-15750).max()),1e-10)
    check('delayed follow-up preserves nadirs',float(abs(pal.nadir_0_182-pal.nadir_0_168).max()),1e-8)
    # Independent raw-event endpoint reconstruction.
    raw=pd.read_csv(Path(os.environ['LOGOS_DATA'])/'synthetic_all.csv')
    panel=pd.read_csv(OUT/'ALL_restart_panel.csv')
    maximum=0.
    for _,e in panel.iterrows():
        q=raw[(raw.ID==e.patient)&(raw.EVID==0)&(raw.TIME>e.restart_time)&(raw.TIME<=e.restart_time+14)]
        maximum=max(maximum,abs(q.DV.min()-e.minimum_ANC))
        check(f'visit count {e.patient}:{e.gap_start}',abs(len(q)-e.n_visits),0)
    check('observed minimum reconstructed from raw events',maximum,1e-10)
    # Check EVERY held-out prediction with centered SVD least squares.
    common=['pre_gap_ANC','restart_dose_mgm2','gap_days','restart_time']
    for label in ['all','two_or_more_visits']:
        pred=pd.read_csv(OUT/f'A_clinical_predictions_{label}.csv')
        maximum=0.
        for pid in pred.patient.unique():
            train=pred[pred.patient!=pid];test=pred[pred.patient==pid]
            for name,cols in [('calendar',common),('state',common+['pre_restart_ANC'])]:
                x=train[cols].to_numpy();mu=x.mean(0);y=train.minimum_ANC.to_numpy()
                beta=np.linalg.lstsq(x-mu,y-y.mean(),rcond=None)[0]
                independent=y.mean()+(test[cols].to_numpy()-mu)@beta
                maximum=max(maximum,float(abs(independent-test['pred_'+name]).max()))
        check(f'all held-out predictions {label}',maximum,1e-8)
    selection=pd.read_csv(OUT/'PSA_selection.csv')
    paired=pd.read_csv(OUT/'strict_primary_PSA_pairs.csv')
    included=set(map(tuple,selection.loc[selection.reason=='Included',['patient','off_start']].to_numpy()))
    actual=set(map(tuple,paired[['patient','off_start']].to_numpy()))
    check('strict PSA selection keys',len(included.symmetric_difference(actual)),0)
    for name in ['A_extension_verification','A_finite_constant_family']:
        record=json.loads((OUT/f'{name}.json').read_text())
        for c in record['checks']:
            check(name+': '+c['check'],c['error'],c['tolerance'])
    from .bootstrap_audit import audit
    independent=audit(OUT)
    for name,record in independent['samples'].items():
        check('all full-training SVD bootstrap losses '+name,record['max_absolute_loss_error'],1e-8)
    result=dict(all_passed=all(c['passed'] for c in checks),checks=checks,
                scope='Computational consistency of synthetic analyses and specified models')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_passed']:
        raise RuntimeError([c for c in checks if not c['passed']])
    print(f'{len(checks)} checks passed')


if __name__=='__main__':main()
