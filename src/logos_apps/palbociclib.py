"""Oral PK and neutropenia simulation with 14-day follow-up."""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def palbo_extended(method='RK45',rtol=2e-9):
 ka=.187*24;kel=57.09/1580*24;lag=.658/24;ktr=4/5.29;base=2.92
 rows=[];traces=[]
 for name in ['21 ON / 7 OFF','Daily redistribution']:
  dose=125 if name.startswith('21') else 93.75
  days=[d for d in range(168) if name=='Daily redistribution' or d%28<21]
  events={d+lag:dose for d in days};bounds=sorted({0.,168.,182.,*events})
  y=np.array([0.,0.,base,base,base,base,base,0.]);extrema=[];parts=[];state168=None
  def fun(t,y):
   gut,central,prol,t1,t2,t3,anc,eliminated=y;C=1000*central/1580
   return [-ka*gut,ka*gut-kel*central,ktr*prol*((1-.0011*C)*(base/anc)**.102-1),ktr*(prol-t1),ktr*(t1-t2),ktr*(t2-t3),ktr*(t3-anc),kel*central]
  for start,end in zip(bounds[:-1],bounds[1:]):
   y[0]+=events.get(start,0)
   so=solve_ivp(fun,(start,end),y,method=method,rtol=rtol,atol=rtol/200,max_step=.05,dense_output=True)
   assert so.success and np.min(so.y[2:7])>0
   ts=np.linspace(start,end,max(3,int((end-start)*20)+1));der=[fun(t,so.sol(t))[6] for t in ts]
   roots=[brentq(lambda t:fun(t,so.sol(t))[6],a,b) for a,b,da,db in zip(ts[:-1],ts[1:],der[:-1],der[1:]) if da*db<0]
   extrema.extend((t,float(so.sol(t)[6])) for t in [start,end,*roots])
   vals=so.sol(ts);parts.append(pd.DataFrame(dict(time_days=ts,ANC=vals[6],C_ug_L=1000*vals[1]/1580,regimen=name)))
   y=so.y[:,-1]
   if end==168:state168=y.copy()
  prior=min((x for x in extrema if x[0]<=168),key=lambda x:x[1]);tail=min((x for x in extrema if x[0]>=168),key=lambda x:x[1]);whole=min(extrema,key=lambda x:x[1])
  rows.append(dict(regimen=name,administered_mg=sum(events.values()),last_dose_time_days=max(events),ANC_at_day168=state168[6],nadir_0_168=prior[1],nadir_day_0_168=prior[0],tail_nadir_ANC=tail[1],tail_nadir_day=tail[0],nadir_0_182=whole[1],nadir_day_0_182=whole[0],ANC_at_day182=y[6],PK_remaining_mg=y[0]+y[1],PK_eliminated_mg=y[7],PK_mass_balance_error=abs(y[0]+y[1]+y[7]-sum(events.values())),method=method))
  traces.append(pd.concat(parts,ignore_index=True))
 return pd.DataFrame(rows),pd.concat(traces,ignore_index=True)
