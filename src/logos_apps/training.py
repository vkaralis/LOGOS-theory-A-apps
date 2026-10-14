"""Bootstrap patient clusters and repeat the complete LOPO learning procedure."""
from pathlib import Path
import os,json
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
OUT=Path(os.environ['LOGOS_OUTPUT']);OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(OUT/'ALL_restart_panel.csv');d['gap_days']=d.restart_time-d.gap_start
common=['pre_gap_ANC','restart_dose_mgm2','gap_days','restart_time'];extra=common+['pre_restart_ANC'];results={}
for label,panel in [('all',d),('two_or_more_visits',d[d.n_visits>=2])]:
 z=panel.dropna(subset=extra+['minimum_ANC']).copy();groups=[g for _,g in z.groupby('patient')];n=len(groups);rng=np.random.default_rng(20261010);losses=[];invalid=0
 # Scaling is a fixed invertible change of basis for unpenalized OLS; it cannot change predictions or select features.
 scale=z[extra].std().to_numpy();center=z[extra].mean().to_numpy();X=np.column_stack([np.ones(len(z)),(z[extra].to_numpy()-center)/scale]);Y=z.minimum_ANC.to_numpy();patients=z.patient.to_numpy();inds=[np.flatnonzero(patients==g.patient.iloc[0]) for g in groups]
 XX=[X[ix].T@X[ix] for ix in inds];XY=[X[ix].T@Y[ix] for ix in inds]
 cols0=np.arange(len(common)+1);cols1=np.arange(len(extra)+1)
 for rep in range(2000):
  multiplicity=np.bincount(rng.integers(n,size=n),minlength=n);tx=sum(count*A for count,A in zip(multiplicity,XX));ty=sum(count*b for count,b in zip(multiplicity,XY));totals=np.zeros(2);good=True
  for j,count in enumerate(multiplicity):
   if not count:continue
   A=tx-count*XX[j];b=ty-count*XY[j];ix=inds[j]
   for k,cols in enumerate([cols0,cols1]):
    mat=A[np.ix_(cols,cols)];rhs=b[cols]
    if np.linalg.matrix_rank(mat)<len(cols):good=False;break
    beta=np.linalg.solve(mat,rhs);pred=X[ix][:,cols]@beta;totals[k]+=count*np.mean((Y[ix]-pred)**2)/n
   if not good:break
  if good:losses.append(dict(replicate=rep,MSE_baseline=float(totals[0]),MSE_state=float(totals[1]),MSE_reduction=float(totals[0]-totals[1])))
  else:invalid+=1
  if (rep+1)%250==0:print(label,'full-training bootstrap',rep+1,flush=True)
 pd.DataFrame(losses).to_csv(OUT/('A_training_bootstrap_'+label+'.csv'),index=False)
 vals=np.array([x['MSE_reduction'] for x in losses]);results[label]=dict(episodes=len(z),patients=n,valid_replicates=len(losses),invalid_replicates=invalid,median_bootstrap_MSE_reduction=float(np.median(vals)),full_training_cluster_interval_95=np.quantile(vals,[.025,.975]).tolist(),positive_bootstrap_fraction=float(np.mean(vals>0)))
(OUT/'A_full_training_uncertainty.json').write_text(json.dumps(dict(seed=20261010,bootstrap_replicates=2000,method='Resample patients; preserve all episodes; refit both OLS models excluding every copy of the held-out patient; summarize multiplicity-weighted patient losses',models=results,interpretation='Internal empirical-population learning uncertainty; not external validation or causal policy effect'),indent=2))
print('Full-training uncertainty complete',flush=True)
