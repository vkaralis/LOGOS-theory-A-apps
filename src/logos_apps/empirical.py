"""Synthetic event analysis. All observations are simulated."""
from pathlib import Path
import argparse, hashlib, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get('LOGOS_DATA', ROOT.parent / 'data'))
SEED = 20261007
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.dpi':300,'svg.fonttype':'none'})
BLUE, ORANGE, GRAY = '#176B91', '#C55A11', '#666666'

def savefig(fig, out, name):
    if name.startswith(('F01','F02','F03','F04','synthetic')):
        fig.suptitle('Synthetic data example', fontsize=12)
    for ext in ('png','svg','pdf'):
        metadata = {
            'png': {'Software': None},
            'svg': {'Date': None, 'Creator': None},
            'pdf': {'Creator': None, 'Producer': None, 'CreationDate': None,
                    'ModDate': None, 'Author': None},
        }[ext]
        fig.savefig(out/'figures'/f'{name}.{ext}',bbox_inches='tight',metadata=metadata)
    plt.close(fig)

def write(df,out,name):
    df.to_csv(out/'tables'/f'{name}.csv',index=False,float_format='%.12g')

def summary(df, col, name, unit, proportion=False, boot=2000):
    z=df[['patient',col]].dropna()
    if len(z)==0:
        return dict(metric=name,unit=unit,n_records=0,n_patients=0,estimate=None,ci_low=None,ci_high=None)
    arrays=[g[col].to_numpy(float) for _,g in z.groupby('patient')]
    rng=np.random.default_rng(SEED)
    fn=np.mean if proportion else np.median
    draws=[fn(np.concatenate([arrays[k] for k in rng.integers(len(arrays),size=len(arrays))])) for _ in range(boot)]
    vals=z[col].to_numpy(float)
    return dict(metric=name,unit=unit,n_records=len(z),n_patients=len(arrays),
                estimate=float(fn(vals)),q25=float(np.quantile(vals,.25)),q75=float(np.quantile(vals,.75)),
                ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),
                estimand='record-weighted proportion' if proportion else 'record-weighted median',
                ci_method='patient-cluster percentile bootstrap; 2000 resamples')

def load_abi():
    return pd.read_csv(RAW/'synthetic_psa.csv').sort_values(['patient','time']).reset_index(drop=True)

def log_slope(g):
    z=g[g.PSA>0]
    return np.polyfit(z.time,np.log(z.PSA),1)[0] if len(z)>=3 and z.time.max()>z.time.min() else np.nan

def abi_analysis(out):
    d=load_abi()
    assert not d.duplicated(['patient','time']).any()
    assert d[['time','PSA','relative_PSA']].notna().all().all()
    assert set(d.on.unique()) <= {0,1} and (d.PSA>0).all()
    episodes=[]; patients=[]; slopes=[]
    for pid,g in d.groupby('patient',sort=True):
        g=g.sort_values('time').reset_index(drop=True)
        first=int(np.flatnonzero(g.on.to_numpy()==1)[0])
        t=g.time.to_numpy(); a=g.on.to_numpy(); psa=g.PSA.to_numpy(); rp=g.relative_PSA.to_numpy()
        dt=np.diff(t[first:]); ondays=float(np.dot(dt,a[first:-1]))
        total=float(t[-1]-t[first])
        # Sensitivity to unknown within-visit transition time: L/R interval assignment and midpoint.
        rightdays=float(np.dot(dt,a[first+1:]))
        transitions=np.flatnonzero(np.diff(a)!=0)+1
        r=psa/rp
        patients.append(dict(patient=pid,cohort=g.cohort.iloc[0],n_measurements=len(g),
                             first_on_time=t[first],post_start_followup_days=total,on_days_left=ondays,
                             off_fraction_left=1-ondays/total,off_fraction_right=1-rightdays/total,
                             off_fraction_midpoint=1-(ondays+rightdays)/2/total,
                             restart_count=int(sum(a[j]==1 and j>first for j in transitions)),
                             normalization_reference=float(np.median(r)),
                             normalization_reference_relative_range=float(np.ptp(r)/np.median(r))))
        for j in transitions:
            if a[j]!=0 or j<=first: continue
            later=[k for k in transitions if k>j and a[k]==1]
            k=later[0] if later else len(g)-1
            complete=bool(later)
            off=g.iloc[j:k+1]
            row=dict(patient=pid,cohort=g.cohort.iloc[0],off_start=t[j],off_end=t[k],
                     off_days=t[k]-t[j],complete_restart=complete,n_off_samples=len(off),
                     PSA_start=psa[j],PSA_end=psa[k],relative_PSA_stop=rp[j],relative_PSA_restart=rp[k],
                     PSA_fold_change=psa[k]/psa[j],log_PSA_change=np.log(psa[k]/psa[j]),
                     off_log_slope=log_slope(off),stop_at_or_below_half=float(rp[j]<=.5),
                     restart_at_or_above_reference=float(rp[k]>=1) if complete else np.nan)
            if complete:
                nextoff=[v for v in transitions if v>k and a[v]==0]
                e=nextoff[0] if nextoff else len(g)-1
                on=g.iloc[k:e+1]
                row['following_on_days']=t[e]-t[k]
                row['following_on_log_slope']=log_slope(on)
                # Fixed landmark: nearest observation to day 60, only at 28-90 days while ON.
                eligible=on[(on.time>=t[k]+28)&(on.time<=t[k]+90)]
                if len(eligible):
                    z=eligible.loc[(eligible.time-(t[k]+60)).abs().idxmin()]
                    row['landmark_days']=z.time-t[k]
                    row['landmark_PSA_fold']=z.PSA/psa[k]
                    row['landmark_PSA_decline']=float(z.PSA<psa[k])
                else:
                    row['landmark_days']=row['landmark_PSA_fold']=row['landmark_PSA_decline']=np.nan
            else:
                row['following_on_days']=row['following_on_log_slope']=np.nan
                row['landmark_days']=row['landmark_PSA_fold']=row['landmark_PSA_decline']=np.nan
            episodes.append(row)
        # Segment descriptions retain both endpoints and require >=3 positive PSA observations.
        bounds=sorted(set([first]+[j for j in transitions if j>first]+[len(g)-1]))
        for start,end in zip(bounds[:-1],bounds[1:]):
            seg=g.iloc[start:end+1]
            slopes.append(dict(patient=pid,cohort=g.cohort.iloc[0],on=int(a[start]),
                               start=t[start],end=t[end],n_samples=len(seg),log_PSA_slope=log_slope(seg)))
    pat=pd.DataFrame(patients);ep=pd.DataFrame(episodes);sl=pd.DataFrame(slopes)
    write(d,out,'S01_abiraterone_observations');write(pat,out,'S02_abiraterone_patients')
    write(ep,out,'S03_abiraterone_off_episodes');write(sl,out,'S04_abiraterone_segment_slopes')
    ad=ep[(ep.cohort=='Adaptive')&ep.complete_restart].copy()
    ad['PSA_increased']= (ad.PSA_fold_change>1).astype(float)
    rows=[]
    for c in ['Adaptive','SOC']:
        z=pat[pat.cohort==c]
        for col,label,unit in [('post_start_followup_days','Follow-up after first recorded ON','days'),
                               ('off_fraction_left','OFF fraction; left-continuous convention','fraction'),
                               ('off_fraction_right','OFF fraction; right convention sensitivity','fraction'),
                               ('restart_count','Recorded restarts','count')]:
            x=summary(z,col,label,unit);x['cohort']=c;rows.append(x)
    for col,label,unit,prop in [('off_days','Completed OFF duration','days',False),
                               ('PSA_fold_change','PSA fold change during OFF','ratio',False),
                               ('off_log_slope','Within-OFF log PSA slope','1/day',False),
                               ('PSA_increased','PSA increased during OFF','fraction',True),
                               ('landmark_PSA_fold','PSA ratio at post-restart landmark','ratio',False),
                               ('landmark_PSA_decline','PSA decline at post-restart landmark','fraction',True),
                               ('stop_at_or_below_half','Stop relative PSA <=0.5','fraction',True),
                               ('restart_at_or_above_reference','Restart relative PSA >=1','fraction',True)]:
        x=summary(ad,col,label,unit,prop);x['cohort']='Adaptive completed OFF';rows.append(x)
    tab=pd.DataFrame(rows);write(tab,out,'T03_abiraterone_summary')
    pair=ad.dropna(subset=['off_log_slope','following_on_log_slope']).copy()
    pair['on_minus_off_log_slope']=pair.following_on_log_slope-pair.off_log_slope
    contrast=summary(pair,'on_minus_off_log_slope','Following ON minus OFF log PSA slope','1/day')
    write(pd.DataFrame([contrast]),out,'T04_abiraterone_paired_slopes')

    fig,ax=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for j,c in enumerate(['Adaptive','SOC']):
        vals=pat.loc[pat.cohort==c,'off_fraction_left']
        ax[0].scatter(np.full(len(vals),j)+np.linspace(-.13,.13,len(vals)),100*vals,color=BLUE if j==0 else GRAY,s=30)
        ax[0].plot([j-.18,j+.18],[100*vals.median()]*2,color='black',lw=2)
    ax[0].set(xticks=[0,1],xticklabels=[f'{c}\n(n={sum(pat.cohort==c)})' for c in ['Adaptive','SOC']],ylabel='Recorded OFF time (%)',title='A  Post-start administration')
    ax[1].hist(ad.off_days,bins=14,color=BLUE,edgecolor='white');ax[1].set(xlabel='Completed OFF duration (days)',ylabel='Episodes',title=f'B  Completed gaps (n={len(ad)})')
    ax[2].scatter(ad.PSA_fold_change,ad.landmark_PSA_fold,color=BLUE,alpha=.6,s=25)
    ax[2].axhline(1,color=GRAY,ls='--');ax[2].axvline(1,color=GRAY,ls='--')
    ax[2].set(xscale='log',yscale='log',xlabel='PSA ratio during OFF',ylabel='PSA ratio after restart',title='C  Rebound and later response')
    savefig(fig,out,'F03_abiraterone_summary')
    # Select four examples by follow-up rank, not by favorable outcomes.
    ids=pat[pat.cohort=='Adaptive'].sort_values(['post_start_followup_days','patient']).patient.to_list()
    ids=[ids[k] for k in np.linspace(0,len(ids)-1,4,dtype=int)]
    fig,axes=plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
    for ax,pid in zip(axes.ravel(),ids):
        g=d[d.patient==pid];p=pat[pat.patient==pid].iloc[0]
        ax.plot(g.time,g.relative_PSA,'o-',color=BLUE,ms=3,lw=1)
        for i in range(len(g)-1):
            if g.iloc[i].on==0 and g.iloc[i].time>=p.first_on_time:
                ax.axvspan(g.iloc[i].time,g.iloc[i+1].time,color=ORANGE,alpha=.15)
        ax.axhline(.5,color=GRAY,ls=':');ax.axhline(1,color=GRAY,ls='--')
        ax.set(title=pid,xlabel='Recorded time (days)',ylabel='Synthetic relative PSA',yscale='log')
    savefig(fig,out,'F04_abiraterone_trajectories')
    return d,pat,ep,tab,contrast

def all_analysis(out):
    d=pd.read_csv(RAW/'synthetic_all.csv')
    assert d.notna().all().all() and not d.duplicated(['ID','TIME','EVID','CMT']).any()
    o=d[d.EVID==0].copy();a=d[d.EVID==1].copy()
    assert (o.CMT==8).all() and (o.MDV==0).all() and (a.CMT==1).all() and (a.MDV==1).all()
    assert (a.AMT>0).all() and (o.DV>=0).all()
    d['patient']=d.ID.astype(int).astype(str);o['patient']=o.ID.astype(int).astype(str);a['patient']=a.ID.astype(int).astype(str)
    patients=[];gaps=[]
    for pid,g in d.groupby('patient',sort=True):
        q=o[o.patient==pid].sort_values('TIME');b=a[a.patient==pid].sort_values('TIME')
        t=b.TIME.to_numpy();amount=b.AMT.to_numpy();dd=np.diff(t)
        patients.append(dict(patient=pid,n_ANC=len(q),n_doses=len(b),first_dose=t[0],last_dose=t[-1],
                             dose_span_days=t[-1]-t[0]+1,followup_days=g.TIME.max()-g.TIME.min(),
                             total_6MP_mg=amount.sum(),mean_dose_per_calendar_day_mg=amount.sum()/(t[-1]-t[0]+1),
                             mean_dose_mg_per_m2=(b.AMT/b.BSA).sum()/(t[-1]-t[0]+1),
                             ANC_median=q.DV.median(),ANC_min=q.DV.min(),
                             observed_fraction_below_05=(q.DV<.5).mean(),
                             observed_fraction_in_05_2=((q.DV>=.5)&(q.DV<=2)).mean(),
                             n_recorded_gaps=int(sum(dd>1.5))))
        for k in np.flatnonzero(dd>1.5):
            start=t[k]+1;end=t[k+1];miss=end-start
            before=q[(q.TIME<=start)&(q.TIME>=start-7)]
            at=q[(q.TIME>=start)&(q.TIME<=end)&(q.TIME>=end-7)]
            after=q[(q.TIME>end)&(q.TIME<=end+14)]
            within=q[(q.TIME>=start)&(q.TIME<=end)]
            row=dict(patient=pid,last_dose_time=t[k],gap_start=start,restart_time=end,gap_days=miss,
                     dose_before_mg=amount[k],dose_restart_mg=amount[k+1],
                     restart_dose_ratio=amount[k+1]/amount[k],n_ANC_within_gap=len(within))
            pre=before.iloc[-1] if len(before) else None
            rest=at.iloc[-1] if len(at) else None
            row['ANC_pre']=pre.DV if pre is not None else np.nan
            row['ANC_pre_time']=pre.TIME if pre is not None else np.nan
            row['ANC_restart']=rest.DV if rest is not None else np.nan
            row['ANC_restart_time']=rest.TIME if rest is not None else np.nan
            row['ANC_post14_min']=after.DV.min() if len(after) else np.nan
            row['ANC_gap_min']=within.DV.min() if len(within) else np.nan
            paired=pre is not None and rest is not None and rest.TIME>pre.TIME
            row['paired_ANC']=paired
            row['ANC_change']=rest.DV-pre.DV if paired else np.nan
            row['ANC_increased']=float(rest.DV>pre.DV) if paired else np.nan
            row['restart_ANC_ge_05']=float(rest.DV>=.5) if rest is not None else np.nan
            row['pre_ANC_below_05']=float(pre.DV<.5) if pre is not None else np.nan
            row['later_ANC_below_restart']=float(after.DV.min()<rest.DV) if len(after) and rest is not None else np.nan
            row['observed_gap_nadir_below_pre']=float(within.DV.min()<pre.DV) if len(within) and pre is not None else np.nan
            gaps.append(row)
    pat=pd.DataFrame(patients);gap=pd.DataFrame(gaps)
    write(pat,out,'S05_ALL_patients');write(gap,out,'S06_ALL_recorded_gaps')
    write(o[['patient','TIME','DV','BSA','WEIGHT','HEIGHT']],out,'S07_ALL_observed_ANC')
    rows=[]
    for min_days in [1,3,7]:
        z=gap[gap.gap_days>=min_days-.01]
        for col,label,unit,prop in [('gap_days','Recorded gap duration','days',False),
                                   ('ANC_pre','Pre-gap ANC','G/L',False),('ANC_restart','Pre-restart ANC','G/L',False),
                                   ('ANC_change','Paired ANC change','G/L',False),('ANC_increased','Paired ANC increase','fraction',True),
                                   ('restart_ANC_ge_05','Pre-restart ANC >=0.5','fraction',True),
                                   ('restart_dose_ratio','Restart / previous dose','ratio',False),
                                   ('observed_gap_nadir_below_pre','Observed lower ANC during gap than before','fraction',True),
                                   ('later_ANC_below_restart','Lower observed ANC within 14 days after restart','fraction',True)]:
            x=summary(z,col,label,unit,prop);x['minimum_gap_days']=min_days;rows.append(x)
    tab=pd.DataFrame(rows);write(tab,out,'T02_ALL_gap_sensitivity')
    severe=gap[(gap.gap_days>=2.99)&(gap.ANC_pre<.5)]
    write(pd.DataFrame([summary(severe,c,n,u,p) for c,n,u,p in [
        ('ANC_change','ANC change after pre-gap ANC <0.5','G/L',False),
        ('restart_ANC_ge_05','Observed recovery to ANC >=0.5','fraction',True)]]),out,'T05_ALL_low_ANC_subgroup')
    # Zero ANC is retained for counts and differences; no log ANC is computed.
    fig,ax=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    ax[0].hist(gap.gap_days,bins=np.arange(.5,36.6,2),color=BLUE,edgecolor='white')
    ax[0].set(xlabel='Inferred days without a dose record',ylabel='Gaps',title='A  Recorded dose gaps')
    z=gap[(gap.gap_days>=2.99)&gap.paired_ANC]
    ax[1].scatter(z.ANC_pre,z.ANC_restart,color=BLUE,s=20,alpha=.5)
    mx=max(z.ANC_pre.max(),z.ANC_restart.max());ax[1].plot([0,mx],[0,mx],color=GRAY,ls='--')
    ax[1].set(xlabel='Pre-gap ANC (G/L)',ylabel='Pre-restart ANC (G/L)',title=f'B  Paired measurements (n={len(z)})')
    w=tab[tab.metric=='Paired ANC change']
    ax[2].errorbar(w.minimum_gap_days,w.estimate,yerr=[w.estimate-w.ci_low,w.ci_high-w.estimate],fmt='o',color=BLUE,capsize=4)
    ax[2].axhline(0,color=GRAY,ls='--');ax[2].set(xlabel='Minimum gap duration (days)',ylabel='Median ANC change (G/L)',title='C  Patient-cluster 95% intervals',xticks=[1,3,7])
    savefig(fig,out,'F01_ALL_gap_recovery')
    # Selection based only on gap count ranks, avoiding outcome-based examples.
    eligible=pat.sort_values(['n_recorded_gaps','patient']);ids=eligible.patient.to_list()
    ids=[ids[k] for k in np.linspace(0,len(ids)-1,4,dtype=int)]
    fig,axes=plt.subplots(4,1,figsize=(11,10),layout='constrained')
    for ax,pid in zip(axes,ids):
        q=o[o.patient==pid];b=a[a.patient==pid];gg=gap[gap.patient==pid]
        ax.plot(q.TIME,q.DV,'o-',color=BLUE,ms=2.5,lw=.8,label='Observed ANC')
        ax.axhspan(.5,2,color=BLUE,alpha=.08);ax.axhline(.5,color=GRAY,ls='--',lw=.7)
        for _,x in gg.iterrows():ax.axvspan(x.gap_start,x.restart_time,color=ORANGE,alpha=.18)
        ax.set(title=f'Synthetic ALL {pid}',ylabel='ANC (G/L)',xlabel='Time (days)')
        twin=ax.twinx();twin.vlines(b.TIME,0,b.AMT,color=GRAY,alpha=.25,lw=.5);twin.set_ylabel('Recorded dose (mg)',color=GRAY)
    savefig(fig,out,'F02_ALL_trajectories')
    return d,pat,gap,tab

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(ROOT/'results'))
    out=Path(ap.parse_args().output);(out/'tables').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    abi,ap,ae,at,contrast=abi_analysis(out);al,lp,lg,lt=all_analysis(out)
    inv=pd.DataFrame([
        dict(dataset='Synthetic ALL',patients=al.ID.nunique(),observations=int((al.EVID==0).sum()),dose_records=int((al.EVID==1).sum()),
             measured_exposure=False,measured_recovery_marker='ANC; host hematologic state',disease_endpoint='Not available',limitations='6MP event file only; no MTX events, CRP, adherence or reasons for holds'),
        dict(dataset='Synthetic PSA Adaptive',patients=int(abi.loc[abi.cohort=='Adaptive','patient'].nunique()),observations=len(abi[abi.cohort=='Adaptive']),dose_records=np.nan,
             measured_exposure=False,measured_recovery_marker='No direct responsiveness marker',disease_endpoint='PSA',limitations='Binary treatment status; not actual dose amounts; nonrandomized'),
        dict(dataset='Synthetic PSA SOC',patients=int(abi.loc[abi.cohort=='SOC','patient'].nunique()),observations=len(abi[abi.cohort=='SOC']),dose_records=np.nan,
             measured_exposure=False,measured_recovery_marker='No direct responsiveness marker',disease_endpoint='PSA',limitations='Synthetic comparator for inventory only')])
    write(inv,out,'T01_dataset_inventory')
    qa={'all_raw_rows':len(al),'all_patients':int(al.ID.nunique()),'all_ANC_zero':int(sum((al.EVID==0)&(al.DV==0))),
        'all_gaps':len(lg),'all_gap_patients':int(lg.patient.nunique()),'all_gaps_min3':int(sum(lg.gap_days>=2.99)),
        'all_paired_min3':int(sum((lg.gap_days>=2.99)&lg.paired_ANC)),
        'abi_adaptive_samples':int(sum(abi.cohort=='Adaptive')),'abi_soc_samples':int(sum(abi.cohort=='SOC')),
        'abi_completed_off_adaptive':int(sum((ae.cohort=='Adaptive')&ae.complete_restart)),
        'abi_followup_normalization_max_relative_range':float(ap.normalization_reference_relative_range.max()),
        'paired_slope_contrast':contrast,'source_SHA256':{},'analysis_script_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bootstrap_seed':SEED,'bootstrap_replicates':2000,
        'limitations':['Gaps in prescribed records do not prove ingestion stopped or identify the cause.',
                       'ANC at sampled visits does not certify full-path safety or tumor benefit.',
                       'No clinical CRP values are included in these fictional records.',
                       'Abiraterone status between visits follows source left-interval convention.',
                       'No survival outcomes are simulated.',
                       'No causal equal-budget empirical comparison or identification of canonical R.']}
    for p in [RAW/'synthetic_all.csv',RAW/'synthetic_psa.csv']:
        qa['source_SHA256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'empirical_summary.json').write_text(json.dumps(qa,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in qa.items() if k not in ['limitations','source_SHA256']},indent=2))

if __name__=='__main__':main()
