"""Synthetic ascertainment, timing, influence and delayed-tail diagnostics."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from .empirical import summary, write, savefig
from .palbociclib import palbo_extended
import matplotlib.pyplot as plt

OUT = Path(os.environ['LOGOS_OUTPUT'])
RAW = Path(os.environ['LOGOS_DATA'])


def save(name, value):
    (OUT/f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def working_fit(x, y, penalty):
    mu, sd = x.mean(0), x.std(0)
    if np.any(sd <= 0):
        raise ValueError('Constant predictor in observation model')
    a = np.column_stack([np.ones(len(x)), (x-mu)/sd])
    def obj(beta):
        eta = a@beta
        return (float(np.logaddexp(0, eta).sum()-y@eta+penalty/2*(beta[1:]@beta[1:])),
                a.T@(expit(eta)-y)+np.r_[0, penalty*beta[1:]])
    res = minimize(obj, np.zeros(a.shape[1]), jac=True, method='BFGS',
                   options={'gtol':1e-8, 'maxiter':3000})
    grad = float(np.max(np.abs(obj(res.x)[1])))
    if grad > 1e-5:
        raise RuntimeError(f'Observation model gradient {grad}')
    return res.x, mu, sd, grad, bool(res.success)


def coefficient_uncertainty(frame, features):
    x=np.column_stack([np.ones(len(frame)),frame[features].to_numpy(float)])
    y=frame.minimum_ANC.to_numpy(float)
    groups=[np.flatnonzero(frame.patient.to_numpy()==p) for p in sorted(frame.patient.unique())]
    def fit(weights):
        a=x*np.sqrt(weights)[:,None];b=y*np.sqrt(weights)
        beta,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
        return float(beta[1]) if rank==x.shape[1] else None
    principal=fit(np.ones(len(frame)))
    if principal is None:
        return dict(estimate=None,reason='Rank deficient',valid_bootstrap=0,failed_bootstrap=2000)
    rng=np.random.default_rng(20261010);draws=[]
    for _ in range(2000):
        counts=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))
        weights=np.zeros(len(frame))
        for count,ix in zip(counts,groups):weights[ix]=count
        value=fit(weights)
        if value is not None:draws.append(value)
    deletions=[]
    for ix in groups:
        weights=np.ones(len(frame));weights[ix]=0
        value=fit(weights)
        if value is not None:deletions.append(value)
    return dict(estimate=principal,ci_low=float(np.quantile(draws,.025)) if draws else None,
                ci_high=float(np.quantile(draws,.975)) if draws else None,
                valid_bootstrap=len(draws),failed_bootstrap=2000-len(draws),
                lopo_min=min(deletions) if deletions else None,lopo_max=max(deletions) if deletions else None)


def main():
    gaps = pd.read_csv(OUT/'tables/S06_ALL_recorded_gaps.csv')
    anc = pd.read_csv(OUT/'tables/S07_ALL_observed_ANC.csv')
    eligible = gaps[(gaps.gap_days >= 2.99) & gaps.paired_ANC].copy()
    rows = []
    for _, e in eligible.iterrows():
        q = anc[(anc.patient == e.patient) & (anc.TIME > e.restart_time)
                & (anc.TIME <= e.restart_time+14)].sort_values('TIME')
        rows.append(dict(patient=e.patient, gap_start=e.gap_start, n_post=len(q),
                         first_lag=float(q.TIME.iloc[0]-e.restart_time) if len(q) else None,
                         min_lag=float(q.loc[q.DV.idxmin(),'TIME']-e.restart_time) if len(q) else None))
    sampling = pd.DataFrame(rows)
    sampling.to_csv(OUT/'sampling_audit.csv', index=False)
    timing = eligible.copy()
    timing['measurement_separation'] = timing.ANC_restart_time-timing.ANC_pre_time
    write(timing, OUT, 'ANC_measurement_timing')
    selection = []
    for label, frame in [('all_gaps', gaps), ('eligible_paired', eligible),
                         ('observed_outcome', sampling[sampling.n_post>0]),
                         ('two_or_more', sampling[sampling.n_post>=2])]:
        selection.append(dict(stage=label, episodes=len(frame), patients=int(frame.patient.nunique())))
    write(pd.DataFrame(selection), OUT, 'ANC_selection')
    z = eligible.merge(sampling, on=['patient','gap_start'], validate='one_to_one')
    cols = ['ANC_pre', 'ANC_restart', 'dose_restart_mg', 'gap_days', 'restart_time']
    x, y, ids = z[cols].to_numpy(float), (z.n_post>=2).to_numpy(float), z.patient.to_numpy()
    full, predictions, coefficients, se_rows = {}, [], [], []
    for name, penalty in [('MLE', 0.), ('ridge_lambda1', 1.)]:
        beta, mu, sd, gradient, success = working_fit(x, y, penalty)
        full[name] = dict(beta=beta.tolist(), mean=mu.tolist(), sd=sd.tolist(),
                          max_gradient=gradient, optimizer_success=success)
        design=np.column_stack([np.ones(len(x)),(x-mu)/sd]);prob=expit(design@beta)
        bread=np.linalg.inv(design.T@((prob*(1-prob))[:,None]*design)+np.diag([0]+[penalty]*len(cols)))
        scores=np.array([design[ids==p].T@(y[ids==p]-prob[ids==p]) for p in np.unique(ids)])
        covariance=bread@(scores.T@scores)@bread
        for term,value,se in zip(['intercept']+cols,beta,np.sqrt(np.maximum(0,np.diag(covariance)))):
            se_rows.append(dict(method=name,term=term,standardized_coefficient=float(value),patient_cluster_SE=float(se)))
        for pid in np.unique(ids):
            train = ids != pid
            b, m, s, grad, ok = working_fit(x[train], y[train], penalty)
            prob = expit(np.column_stack([np.ones((~train).sum()), (x[~train]-m)/s])@b)
            base = float((y[train].sum()+.5)/(len(y[train])+1))
            for ix, q in zip(np.flatnonzero(~train), prob):
                predictions.append(dict(method=name, patient=pid, gap_start=z.iloc[ix].gap_start,
                                        event=y[ix], probability=q, baseline=base))
            coefficients.append(dict(method=name, patient=pid, gradient=grad, success=ok,
                                     **dict(zip(['intercept']+cols,b.tolist()))))
    pred = pd.DataFrame(predictions)
    pred['brier'] = (pred.event-pred.probability)**2
    pred['baseline_brier'] = (pred.event-pred.baseline)**2
    pred.to_csv(OUT/'observation_predictions.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'observation_coefficients.csv',index=False)
    write(pd.DataFrame(se_rows),OUT,'observation_cluster_coefficients')
    save('observation_process',dict(synthetic=True, endpoint='At least two recorded visits in (restart,restart+14]',
                                   predictors=cols, full_fits=full,
                                   metrics=pred.groupby(['method','patient'])[['brier','baseline_brier']].mean().groupby('method').mean().to_dict('index')))
    panel = pd.read_csv(OUT/'ALL_restart_panel.csv')
    extended = panel.merge(sampling[['patient','gap_start','first_lag','min_lag']],on=['patient','gap_start'])
    extended.to_csv(OUT/'restart_safety_extended.csv',index=False)
    sensitivity = []
    for label, frame in [('all',extended), ('two_or_more',extended[extended.n_visits>=2]),
                          ('three_or_more',extended[extended.n_visits>=3]),
                          ('near_restart',extended[extended.restart_lag<=1.5])]:
        for features, spec in [(['pre_restart_ANC','pre_gap_ANC','restart_dose_mgm2'],'primary'),
                               (['pre_restart_ANC','pre_gap_ANC','restart_dose_ratio','dose14_mgm2'],'future_amount'),
                               (['pre_restart_ANC','pre_gap_ANC','restart_dose_mgm2','n_visits','first_lag'],'sampling_adjusted')]:
            frame = frame.dropna(subset=features+['minimum_ANC'])
            X = np.column_stack([np.ones(len(frame)),frame[features].to_numpy(float)])
            value = None
            if len(frame) and np.linalg.matrix_rank(X)==X.shape[1]:
                value = float(np.linalg.lstsq(X,frame.minimum_ANC,rcond=None)[0][1])
            uncertainty=coefficient_uncertainty(frame,features) if len(frame) else dict(estimate=None,reason='No complete records')
            sensitivity.append(dict(sample=label, specification=spec, episodes=len(frame),
                                    patients=int(frame.patient.nunique()), **uncertainty))
    write(pd.DataFrame(sensitivity),OUT,'ANC_specification_sensitivity')
    strata = []
    predictions=pd.read_csv(OUT/'A_clinical_predictions_all.csv')
    for count, frame in predictions.groupby('n_visits'):
        strata.append(dict(visits=int(count), episodes=len(frame), patients=int(frame.patient.nunique()),
                           observed_minimum_median=float(frame.minimum_ANC.median()),
                           baseline_RMSE=float(np.sqrt(frame.groupby('patient').sq_calendar.mean().mean())),
                           state_RMSE=float(np.sqrt(frame.groupby('patient').sq_state.mean().mean()))))
    write(pd.DataFrame(strata),OUT,'ANC_visit_strata')
    threshold_rows=[];weight_rows=[]
    for threshold in [2,3,5,7]:
        frame=gaps[(gaps.gap_days>=threshold-.01)&gaps.paired_ANC]
        row=summary(frame,'ANC_change',f'Gap at least {threshold} days','G/L')
        threshold_rows.append(row)
    write(pd.DataFrame(threshold_rows),OUT,'ANC_gap_thresholds')
    for label,frame in [('paired',eligible),('low_baseline',eligible[eligible.ANC_pre<.5])]:
        patients=frame.groupby('patient').ANC_change.median().reset_index()
        row=summary(patients,'ANC_change',label,'G/L');row['estimand']='Median of patient medians'
        weight_rows.append(row)
    write(pd.DataFrame(weight_rows),OUT,'ANC_patient_weighted_changes')
    # Landmark selection always distinguishes the treatment switch endpoint.
    obs = pd.read_csv(RAW/'synthetic_psa.csv')
    off = pd.read_csv(OUT/'tables/S03_abiraterone_off_episodes.csv')
    landmark, flow = [], []
    for _, e in off[(off.cohort=='Adaptive') & off.complete_restart].iterrows():
        q = obs[obs.patient==e.patient].sort_values('time').reset_index(drop=True)
        k = int(np.flatnonzero(np.isclose(q.time,e.off_end))[0])
        nextoff = q[(q.index>k)&(q.on==0)]
        stop = float(nextoff.time.iloc[0]) if len(nextoff) else np.inf
        for lo,hi,strict in [(28,90,False),(28,90,True),(42,78,False),(14,120,False)]:
            cand = q[(q.time>=e.off_end+lo)&(q.time<=e.off_end+hi)&(q.time<=stop)]
            if strict: cand = cand[cand.time<stop]
            if len(cand):
                visit=cand.loc[(cand.time-e.off_end-60).abs().idxmin()]
                landmark.append(dict(patient=e.patient,off_start=e.off_start,window=f'{lo}-{hi}',
                                     strict=strict,value=visit.PSA/e.PSA_end,lag=visit.time-e.off_end,
                                     at_switch=bool(visit.time==stop)))
        prev=q[(q.time<e.off_start)&(q.on==1)]
        # Keep the contiguous ON block immediately preceding this OFF start.
        previous_off=q[(q.time<e.off_start)&(q.on==0)]
        if len(previous_off): prev=prev[prev.time>previous_off.time.max()]
        post=q[(q.time>=e.off_end)&(q.time<stop)&(q.on==1)]
        reason='Included' if len(prev)>=3 and len(post)>=3 else ('Previous ON has fewer than 3 positive values' if len(prev)<3 else 'Subsequent ON has fewer than 3 positive values')
        flow.append(dict(patient=e.patient,off_start=e.off_start,previous_n=len(prev),subsequent_n=len(post),reason=reason))
    lm=pd.DataFrame(landmark);lm.to_csv(OUT/'PSA_landmarks.csv',index=False)
    pd.DataFrame(flow).to_csv(OUT/'PSA_selection.csv',index=False)
    baseline=[]
    targets=pd.read_csv(OUT/'low_ANC_targets.csv')
    matching=OUT/'within_patient_primary.csv'
    matched=pd.read_csv(matching) if matching.stat().st_size>1 else pd.DataFrame(columns=['target_id'])
    for flag,label in [(True,'included'),(False,'excluded')]:
        group=targets[targets.target_id.isin(matched.target_id)==flag]
        for column in ['initial','sep','intensity','position','bsa']:
            baseline.append(dict(dataset='ALL',selection=label,variable=column,n=len(group),median=float(group[column].median()) if len(group) else None))
        selected=pd.DataFrame(flow)
        keys=selected.loc[(selected.reason=='Included')==flag,['patient','off_start']]
        group=off.merge(keys,on=['patient','off_start'],validate='one_to_one')
        for column in ['off_days','PSA_end','PSA_start']:
            baseline.append(dict(dataset='PSA',selection=label,variable=column,n=len(group),median=float(group[column].median()) if len(group) else None))
    write(pd.DataFrame(baseline),OUT,'included_excluded_baseline')
    summaries=[]
    for (window,strict),frame in lm.groupby(['window','strict']):
        row=summary(frame,'value',f'{window}; strict={strict}','ratio');summaries.append(row)
    write(pd.DataFrame(summaries),OUT,'PSA_landmark_sensitivity')
    influence=[]
    strict=lm[(lm.window=='28-90')&lm.strict]
    completed=off[(off.cohort=='Adaptive')&off.complete_restart]
    for label,frame,column in [('low_baseline_ANC',eligible[eligible.ANC_pre<.5],'ANC_change'),
                               ('OFF_duration',completed,'off_days'),('OFF_fold',completed,'PSA_fold_change'),
                               ('OFF_slope',completed,'off_log_slope'),('strict_landmark',strict,'value')]:
        frame=frame.dropna(subset=[column]);pids=frame.patient.unique()
        estimates=[float(frame.loc[frame.patient!=pid,column].median()) for pid in pids if (frame.patient!=pid).any()]
        influence.append(dict(endpoint=label,episodes=len(frame),patients=len(pids),
                              median=float(frame[column].median()) if len(frame) else None,
                              lopo_min=min(estimates) if estimates else None,lopo_max=max(estimates) if estimates else None))
    write(pd.DataFrame(influence),OUT,'retained_median_influence')
    pal,trace=palbo_extended();ind,_=palbo_extended('DOP853',2e-11)
    write(pal,OUT,'palbociclib_tail');write(ind,OUT,'palbociclib_tail_independent')
    write(trace,OUT,'palbociclib_tail_trace')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].scatter(panel.pre_restart_ANC,panel.minimum_ANC,s=12,alpha=.5)
    axes[0].set(xlabel='Synthetic pre-restart ANC (G/L)',ylabel='Synthetic observed minimum ANC (G/L)')
    for name,g in trace.groupby('regimen'): axes[1].plot(g.time_days,g.ANC,label=name)
    axes[1].axvspan(168,182,color='gray',alpha=.15)
    axes[1].set(xlabel='Time (days)',ylabel='Model ANC (G/L)');axes[1].legend()
    savefig(fig,OUT,'synthetic_state_and_model_tail')
    # Separate clinical and mechanistic figures so the endpoints stay explicit.
    paired=pd.read_csv(OUT/'strict_primary_PSA_pairs.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].scatter(panel.pre_restart_ANC,panel.minimum_ANC,s=12,alpha=.5)
    axes[0].set(xlabel='Pre-restart ANC (G/L)',ylabel='Observed minimum ANC (G/L)')
    for _,e in paired.iterrows():axes[1].plot([0,1],[e.previous_slope,e.subsequent_slope],alpha=.3,color='#176B91')
    axes[1].set(xticks=[0,1],xticklabels=['Previous ON','Subsequent ON'],ylabel='Log PSA slope (1/day)')
    savefig(fig,OUT,'synthetic_clinical_links')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,title,stages in [(axes[0],'ANC ascertainment',selection),
                             (axes[1],'Strict PSA selection',[dict(stage='Completed OFF',episodes=len(flow)),dict(stage='Included',episodes=int((pd.DataFrame(flow).reason=='Included').sum()))])]:
        ax.barh([s['stage'] for s in stages],[s['episodes'] for s in stages],color='#176B91')
        ax.set(title=title,xlabel='Synthetic episodes')
    savefig(fig,OUT,'synthetic_selection_flow')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ft=pd.read_csv(OUT/'tables/S11_finite_trajectories.csv')
    for name,g in ft.groupby('regimen'):axes[0].plot(g.time,g.D,label=name)
    axes[0].axvspan(20,25,color='gray',alpha=.15)
    axes[0].set(xlabel='Normalized time',ylabel='Disease burden',yscale='log');axes[0].legend()
    for name,g in trace.groupby('regimen'):axes[1].plot(g.time_days,g.ANC,label=name)
    axes[1].axvspan(168,182,color='gray',alpha=.15)
    axes[1].set(xlabel='Time (days)',ylabel='Model ANC (G/L)');axes[1].legend()
    savefig(fig,OUT,'matched_model_endpoints')


if __name__=='__main__':main()
