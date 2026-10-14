"""Common initialization and conservative full-constant-family upper bound."""
from pathlib import Path
import os,sys,json
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar,brentq
from . import models
OUT=Path(os.environ['LOGOS_OUTPUT']);OUT.mkdir(parents=True,exist_ok=True)
p=models.P;H=20.;F=25.;N=4096
u=np.linspace(0,1,N+1);y=np.r_[np.ones(len(u)),np.zeros(len(u))]
for lo,hi in [(0,H),(H,F)]:
 def rhs(t,y):
  C=u*(-np.expm1(-4*t)) if lo==0 else u*(-np.expm1(-4*H))*np.exp(-4*(t-H));E,sr,dr,_,_=models.coeff(C,p);R=y[:len(u)];return np.r_[sr-dr*R,E*R-.08*(1-E)]
 sol=solve_ivp(rhs,(lo,hi),y,method='DOP853',rtol=2e-10,atol=1e-12,max_step=.1);assert sol.success;y=sol.y[:,-1]
values=y[len(u):]/F
# For all u in [0,1]: |C_u|<=1, |sr_C|<=r*n/chi, |E_C|<=1.
# dr >= min(r/2, alpha*chi/(1+chi)) > 0, splitting at C=chi.
damping=min(p['r']/2,p['alpha']*p['chi']/(1+p['chi']));R_sensitivity=(p['r']*p['n']/p['chi']+p['alpha'])/damping
L=p['eta']+p['rho']+p['eta']*R_sensitivity
upper=float(values.max()+L/(2*N)+1e-7)
def constant(control,method='DOP853',rtol=2e-11):
 y=np.array([1.,0.,0.]);peak=0.
 for lo,hi in [(0,H),(H,F)]:
  def concentration(t):return control*(-np.expm1(-4*t)) if lo==0 else control*(-np.expm1(-4*H))*np.exp(-4*(t-H))
  def rhs(t,y):
   E,sr,dr,sz,dz=models.coeff(np.array([concentration(t)]),p);return [sr[0]-dr[0]*y[0],sz[0]-dz[0]*y[1],E[0]*y[0]-.08*(1-E[0])]
  sol=solve_ivp(rhs,(lo,hi),y,method=method,rtol=rtol,atol=rtol/100,max_step=.1,dense_output=True);assert sol.success
  times=np.linspace(lo,hi,401);der=np.array([rhs(t,sol.sol(t))[1] for t in times]);roots=[brentq(lambda t:rhs(t,sol.sol(t))[1],a,b) for a,b,da,db in zip(times[:-1],times[1:],der[:-1],der[1:]) if da*db<0];peak=max(peak,float(np.max(sol.sol([lo,hi]+roots)[1])));y=sol.y[:,-1]
 Cend=concentration(F);zeq=p['beta']/p['pi']*Cend/(1+Cend)*(1+(Cend/p['chiZ'])**2)
 return dict(input=float(control),amount=H*control,J=float(y[2]/F),peak_Z=peak,tail_state_Z=float(y[1]),tail_equilibrium_Z=float(zeq),tail_declining=bool(y[1]>=zeq))
ix=int(np.argmax(values));left=u[max(0,ix-1)];right=u[min(N,ix+1)];opt=minimize_scalar(lambda v:-constant(v)['J'],bounds=(left,right),method='bounded',options={'xatol':1e-11});best=constant(opt.x);ind=constant(opt.x,'Radau',2e-12)
period=H/12;f=.2;segments=sorted([(j*period,j*period+f*period,5.) for j in range(12)]+[(j*period+f*period,(j+1)*period,0.) for j in range(12)])+[(H,F,0.)]
def cycle(method='DOP853',rtol=2e-11):
 y=np.array([0.,1.,0.,0.]);peak=0.
 for lo,hi,control in segments:
  def rhs(t,y):
   E,sr,dr,sz,dz=models.coeff(np.array([y[0]]),p);return [4*(control-y[0]),sr[0]-dr[0]*y[1],sz[0]-dz[0]*y[2],E[0]*y[1]-.08*(1-E[0])]
  sol=solve_ivp(rhs,(lo,hi),y,method=method,rtol=rtol,atol=rtol/100,max_step=.03,dense_output=True);assert sol.success
  times=np.linspace(lo,hi,101);der=np.array([rhs(t,sol.sol(t))[2] for t in times]);roots=[brentq(lambda t:rhs(t,sol.sol(t))[2],a,b) for a,b,da,db in zip(times[:-1],times[1:],der[:-1],der[1:]) if da*db<0];peak=max(peak,float(np.max(sol.sol([lo,hi]+roots)[2])));y=sol.y[:,-1]
 zeq=p['beta']/p['pi']*y[0]/(1+y[0])*(1+(y[0]/p['chiZ'])**2)
 return dict(period=period,duty=f,amount=20.,J=float(y[3]/F),peak_Z=peak,tail_state_Z=float(y[2]),tail_equilibrium_Z=float(zeq),tail_declining=bool(y[2]>=zeq))
c=cycle();c2=cycle('Radau',2e-12)
checks=[dict(check='constant independent Radau objective',error=abs(best['J']-ind['J']),tolerance=1e-8),dict(check='cycle independent Radau objective',error=abs(c['J']-c2['J']),tolerance=1e-8),dict(check='cycle full-path injury',error=max(0,c['peak_Z']-.22),tolerance=1e-9),dict(check='constant below conservative upper bound',error=max(0,best['J']-upper),tolerance=1e-10),dict(check='strict lower margin positive',error=max(0,1e-5-(c['J']-upper)),tolerance=0),dict(check='declining complete injury tails',error=float(not(best['tail_declining'] and c['tail_declining'])),tolerance=0)]
for v in checks:v['passed']=bool(v['error']<=v['tolerance'])
result=dict(initial_state={'C':0,'R':1,'Z':0,'relative_burden':1},administration_horizon=H,outcome_horizon=F,available_amount_cap=20,peak_input_cap=5,injury_cap=.22,constant_family='Every constant input u in [0,1] until H; zero thereafter. Upper bound relaxes the injury constraint, hence also bounds every safe member.',best_constant=best,constant_upper_bound=upper,constant_derivative_absolute_bound=L,grid_intervals=N,numerical_integration_allowance=1e-7,cycle=c,strict_numeric_lower_margin=c['J']-upper,realized_amounts_differ=True,checks=checks,all_passed=all(v['passed'] for v in checks))
(OUT/'A_finite_constant_family.json').write_text(json.dumps(result,indent=2));assert result['all_passed'];print('Finite common-state proof complete',c['J']-upper,flush=True)
