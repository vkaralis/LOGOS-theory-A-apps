"""Synthetic matched ANC and strict adjacent-ON PSA analyses."""
from pathlib import Path
import json,hashlib,platform,warnings
import numpy as np,pandas as pd,scipy
from scipy.stats import spearmanr
warnings.filterwarnings('ignore',category=RuntimeWarning)
import os
A=Path(os.environ['LOGOS_OUTPUT']);R=A
RAW=Path(os.environ['LOGOS_DATA'])
g=pd.read_csv(A/'tables/S06_ALL_recorded_gaps.csv')
anc=pd.read_csv(A/'tables/S07_ALL_observed_ANC.csv')
raw=pd.read_csv(RAW/'synthetic_all.csv')
ab=pd.read_csv(RAW/'synthetic_psa.csv')
off=pd.read_csv(A/'tables/S03_abiraterone_off_episodes.csv')
doses=raw[(raw.EVID==1)&(raw.MDV==1)&(raw.CMT==1)].copy()
D={p:r.sort_values('TIME') for p,r in doses.groupby('ID')};N={p:r.sort_values('TIME') for p,r in anc.groupby('patient')}
base=g[(g.gap_days>=2.99)&g.paired_ANC.astype(bool)].copy()
def bsa(p,t):
 a=N[p];r=a[np.isclose(a.TIME,t,atol=1e-6)];return float(r.iloc[0].BSA) if len(r) else np.nan
def feature(p,t,v,sep):
 dd=D[p];B=bsa(p,t);m=dd[(dd.TIME>=t-7)&(dd.TIME<t)].AMT.sum()/7/B
 return dict(patient=p,t=t,initial=float(v),sep=float(sep),intensity=float(m),position=float(t-dd.TIME.min()),bsa=B)
def continuous(p,t,z):
 ts=D[p].TIME.to_numpy();j=np.searchsorted(ts,t,side='right')-1;k=np.searchsorted(ts,z,side='left')
 return j>=0 and k<len(ts) and np.all(np.diff(ts[j:k+1])<=1.5)
controls=[]
for p,a in N.items():
 rr=a.to_dict('records')
 for i in range(len(rr)-1):
  x,y=rr[i:i+2];sep=y['TIME']-x['TIME']
  if x['DV']<.5 and 1<=sep<=25 and continuous(p,x['TIME'],y['TIME']):
   f=feature(p,x['TIME'],x['DV'],sep);f.update(control_id=f'{p}:{i}',end=y['TIME'],final=y['DV'],change=y['DV']-x['DV']);controls.append(f)
target=[]
for i,x in base[base.ANC_pre<.5].iterrows():
 f=feature(x.patient,x.ANC_pre_time,x.ANC_pre,x.ANC_restart_time-x.ANC_pre_time)
 f.update(gap_start=x.gap_start,final=x.ANC_restart,change=x.ANC_restart-x.ANC_pre,target_id=int(i));target.append(f)
C=pd.DataFrame(controls);T=pd.DataFrame(target);C.to_csv(A/'no_gap_candidates.csv',index=False);T.to_csv(A/'low_ANC_targets.csv',index=False)
def match(window=3,same=True):
 options={}
 for t in target:
  opts=[]
  for c in controls:
   if (same and c['patient']!=t['patient']) or min(c['intensity'],t['intensity'])<=0 or not np.isfinite(c['intensity']+t['intensity']):continue
   dif=[c['initial']-t['initial'],c['sep']-t['sep'],np.log(c['intensity']/t['intensity']),c['position']-t['position']]
   scales=[.2,window,np.log(2),180]
   if any(abs(d)>s+1e-8 for d,s in zip(dif,scales)):continue
   score=sum((d/s)**2 for d,s in zip(dif,scales));opts.append((score,c,dif))
  options[t['target_id']]=sorted(opts,key=lambda x:(x[0],x[1]['patient'],x[1]['t']))
 used=set();out=[]
 for t in sorted(target,key=lambda t:(len(options[t['target_id']]),t['patient'],t['gap_start'])):
  pick=next((x for x in options[t['target_id']] if x[1]['control_id'] not in used),None)
  if pick is None:continue
  score,c,dif=pick;used.add(c['control_id'])
  r={'patient':t['patient'],'control_patient':c['patient'],'gap_start':t['gap_start'],'target_id':t['target_id'],'control_id':c['control_id'],'score':score,'gap_change':t['change'],'control_change':c['change'],'difference_in_change':t['change']-c['change']}
  for k in ['t','initial','sep','intensity','position','final']:r['gap_'+k]=t[k];r['control_'+k]=c[k]
  out.append(r)
 return pd.DataFrame(out)
rng=np.random.default_rng(20261009)
def bootstrap(df,fun):
 if len(df)==0:return {'estimate':None,'ci_low':None,'ci_high':None,'valid_bootstrap':0,'failed_bootstrap':2000,'lopo_min':None,'lopo_max':None,'n':0,'patients':0}
 try: principal=float(fun(df))
 except (ValueError,np.linalg.LinAlgError,ZeroDivisionError):
  return dict(estimate=None,ci_low=None,ci_high=None,valid_bootstrap=0,failed_bootstrap=2000,lopo_min=None,lopo_max=None,n=len(df),patients=int(df.patient.nunique()),reason='Rank deficient or undefined estimate')
 pats=df.patient.unique();groups=[df[df.patient==p] for p in pats];vals=[]
 for _ in range(2000):
  ix=rng.integers(len(groups),size=len(groups));r=pd.concat([groups[i].assign(patient=f'bootstrap_cluster_{j}') for j,i in enumerate(ix)],ignore_index=True)
  try:v=float(fun(r))
  except (ValueError,np.linalg.LinAlgError,ZeroDivisionError):v=np.nan
  if np.isfinite(v):vals.append(v)
 lopo=[]
 for p in pats:
  try:v=float(fun(df[df.patient!=p]));
  except (ValueError,np.linalg.LinAlgError,ZeroDivisionError):v=np.nan
  if np.isfinite(v):lopo.append(v)
 return {'estimate':float(fun(df)),'ci_low':float(np.percentile(vals,2.5)) if vals else None,'ci_high':float(np.percentile(vals,97.5)) if vals else None,'valid_bootstrap':len(vals),'failed_bootstrap':2000-len(vals),'lopo_min':min(lopo) if lopo else None,'lopo_max':max(lopo) if lopo else None,'n':len(df),'patients':len(pats)}
result={'match_candidates':len(C),'candidate_patients':int(C.patient.nunique()),'target_n':len(T),'target_patients':int(T.patient.nunique()),'matched':{}}
for label,window,same in [('within_patient_primary',3,True),('within_patient_7day',7,True),('cross_patient_secondary',3,False)]:
 m=match(window,same);m.to_csv(A/(label+'.csv'),index=False)
 z=bootstrap(m,lambda d:np.median(d.difference_in_change));z.update(unmatched=len(T)-len(m))
 if len(m):
  z.update(gap_change_median=float(m.gap_change.median()),control_change_median=float(m.control_change.median()),same_patient_matches=int((m.patient==m.control_patient).sum()))
  z['median_covariate_differences']={k:float(np.median(m['gap_'+k]-m['control_'+k])) for k in ['initial','sep','intensity','position']}
 result['matched'][label]=z
# Cross-patient resampling is target-patient only; shared control patients violate cluster independence. Flag it as a descriptive sensitivity, not a reliable clustered CI.
result['matched']['cross_patient_secondary']['cluster_limit']='Resampling is by target patient; cross-patient control dependence is not fully captured. Interval is not used for inference.'
safe=[]
for i,x in base.iterrows():
 p=x.patient;aa=N[p];aa=aa[(aa.TIME>x.restart_time)&(aa.TIME<=x.restart_time+14)]
 if len(aa)==0:continue
 B=bsa(p,x.ANC_restart_time);dd=D[p];dose14=dd[(dd.TIME>=x.restart_time)&(dd.TIME<x.restart_time+14)].AMT.sum()
 safe.append({'patient':p,'gap_start':x.gap_start,'restart_time':x.restart_time,'minimum_ANC':float(aa.DV.min()),'pre_restart_ANC':x.ANC_restart,'pre_gap_ANC':x.ANC_pre,'restart_dose_mgm2':x.dose_restart_mg/B,'dose14_mgm2':dose14/B,'restart_dose_ratio':x.restart_dose_ratio,'restart_lag':x.restart_time-x.ANC_restart_time,'n_visits':len(aa)})
S=pd.DataFrame(safe).dropna();S.to_csv(A/'restart_safety.csv',index=False)
S.to_csv(A/'ALL_restart_panel.csv',index=False)
def coef(d,extra=False,weighted=False):
 cols=['pre_restart_ANC','pre_gap_ANC','restart_dose_mgm2']+(['dose14_mgm2'] if extra else [])
 X=np.column_stack([np.ones(len(d)),d[cols].to_numpy(float)]);y=d.minimum_ANC.to_numpy(float)
 if weighted:
  weights=1/d.groupby('patient').patient.transform('size').to_numpy(float);X=X*np.sqrt(weights[:,None]);y=y*np.sqrt(weights)
 if np.linalg.matrix_rank(X)<X.shape[1]:raise ValueError('rank deficient')
 return np.linalg.lstsq(X,y,rcond=None)[0][1]
result['safety']={}
for label,d,extra,weighted in [('adjusted_primary',S,False,False),('future_amount_secondary',S,True,False),('patient_weighted',S,False,True),('near_restart',S[S.restart_lag<=1.5],False,False)]:result['safety'][label]=bootstrap(d,lambda r:coef(r,extra,weighted))
result['safety']['spearman']=bootstrap(S,lambda d:spearmanr(d.pre_restart_ANC,d.minimum_ANC).statistic)
X=np.column_stack([np.ones(len(S)),S[['pre_restart_ANC','pre_gap_ANC','restart_dose_mgm2']].to_numpy(float)]);y=S.minimum_ANC.to_numpy(float);beta=np.linalg.lstsq(X,y,rcond=None)[0]
result['safety']['coefficients']=dict(zip(['intercept','pre_restart_ANC','pre_gap_ANC','restart_dose_mgm2'],map(float,beta)))
result['safety']['minimum_range']=[float(S.minimum_ANC.min()),float(S.minimum_ANC.max())]
# Strict ON segments from original observed statuses. Include ON start and exclude OFF endpoint.
obs={p:r.sort_values('time').reset_index(drop=True) for p,r in ab[ab.cohort=='Adaptive'].groupby('patient')}
segments={}
for p,a in obs.items():
 seg=[];i=0
 while i<len(a):
  if int(a.iloc[i].on)!=1:i+=1;continue
  start=i;j=i+1
  while j<len(a) and int(a.iloc[j].on)==1:j+=1
  seg.append({'start':float(a.iloc[start].time),'end':float(a.iloc[j].time) if j<len(a) else float(a.iloc[-1].time),'has_off':j<len(a),'strict':a.iloc[start:j],'inclusive':a.iloc[start:min(j+1,len(a))]});i=j
 segments[p]=seg
def psa_pairs(strict=True,minn=3):
 out=[]
 for _,e in off[(off.cohort=='Adaptive')&off.complete_restart.astype(bool)].iterrows():
  seg=segments[e.patient];prev=next((s for s in seg if s['has_off'] and abs(s['end']-e.off_start)<1e-6),None);nxt=next((s for s in seg if abs(s['start']-e.off_end)<1e-6),None)
  if prev is None or nxt is None:continue
  slope=[];detail={}
  for label,s in [('previous',prev),('subsequent',nxt)]:
   z=s['strict' if strict else 'inclusive'];z=z[z.PSA>0]
   if len(z)<minn or z.time.max()<=z.time.min():break
   v=float(np.polyfit(z.time.to_numpy(float)-z.time.min(),np.log(z.PSA.to_numpy(float)),1)[0]);slope.append(v)
   detail.update({label+'_n':len(z),label+'_days':float(z.time.max()-z.time.min()),label+'_baseline_PSA':float(z.iloc[0].PSA),label+'_slope':v})
  if len(slope)!=2:continue
  out.append(dict(patient=e.patient,off_start=e.off_start,restart=e.off_end,off_days=e.off_days,PSA_restart=e.PSA_end,slope_difference=slope[1]-slope[0],**detail))
 return pd.DataFrame(out)
result['PSA']={}
for label,strict,minn in [('strict_primary',True,3),('switch_inclusive',False,3),('strict_four',True,4)]:
 P=psa_pairs(strict,minn);P.to_csv(A/(label+'_PSA_pairs.csv'),index=False)
 result['PSA'][label]=bootstrap(P,lambda d:np.median(d.slope_difference))
 if len(P):
  result['PSA'][label]['medians']={c:float(P[c].median()) for c in ['previous_slope','subsequent_slope','previous_n','subsequent_n','previous_days','subsequent_days','previous_baseline_PSA','subsequent_baseline_PSA','off_days']}
 if label=='strict_primary' and len(P):
  result['PSA']['patient_weighted']=bootstrap(P,lambda d:np.median(d.groupby('patient').slope_difference.median()))
  if len(P)>=8 and P.patient.nunique()>=4:
   result['PSA']['OFF_association']=bootstrap(P,lambda d:spearmanr(d.off_days,d.slope_difference).statistic)
   result['PSA']['restart_PSA_association']=bootstrap(P,lambda d:spearmanr(np.log(d.PSA_restart),d.slope_difference).statistic)
result['environment']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'seed':20261009,'bootstrap_replicates':2000}
(A/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2,allow_nan=False))
