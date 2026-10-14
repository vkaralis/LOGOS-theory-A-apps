"""Canonical Theory A and published-parameter palbociclib illustrations.
These simulations are not fitted to the empirical datasets.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .empirical import savefig,write,BLUE,ORANGE,GRAY,ROOT

P=dict(kappa=4.,alpha=2.,pi=.7,beta=.2,eta=1.,rho=.08,chi=.35,chiZ=1.,n=4.,r=1.,gate='hill',control='full')

def coeff(C,p):
    E=C/(1+C)
    if p['gate']=='hill': W=1/(1+(C/p['chi'])**p['n'])
    else: W=np.exp(-np.log(2)*(C/p['chi'])**(p['n']/(2*np.log(2))))
    sr=p['r']*W
    if p['control']=='recovery_disabled':sr=np.zeros_like(C)
    dr=sr+p['alpha']*E
    wz=np.ones_like(C) if p['control']=='ordinary_repair' else 1/(1+(C/p['chiZ'])**2)
    return E,sr,dr,p['beta']*E,p['pi']*wz

def screen(T,f,p=P,N=256):
    T,f=np.broadcast_arrays(np.atleast_1d(T).astype(float),np.atleast_1d(f).astype(float))
    amp=1/f;on=T*f;off=T-on;k=p['kappa']
    C0=amp*(-np.expm1(-k*on))*np.exp(-k*off)/(-np.expm1(-k*T))
    Cswitch=amp+(C0-amp)*np.exp(-k*on)
    AR=np.ones_like(T);BR=np.zeros_like(T);AZ=np.ones_like(T);BZ=np.zeros_like(T)
    segments=[]
    for duration,initial,u in [(on,C0,amp),(off,Cswitch,0)]:
        dt=duration/N
        uvec=np.broadcast_to(u,T.shape)
        c=uvec[:,None]+(initial-uvec)[:,None]*np.exp(-k*dt[:,None]*(np.arange(N)+.5))
        E,sr,dr,sz,dz=coeff(c,p)
        rr=np.exp(-dr*dt[:,None]);rz=np.exp(-dz*dt[:,None])
        br=sr*(-np.expm1(-dr*dt[:,None]))/dr
        bz=sz*(-np.expm1(-dz*dt[:,None]))/dz
        segments.append((dt,E,sr,dr,sz,dz,rr,rz,br,bz))
        for j in range(N):
            AR*=rr[:,j];BR=rr[:,j]*BR+br[:,j]
            AZ*=rz[:,j];BZ=rz[:,j]*BZ+bz[:,j]
    R=BR/(1-AR);Z=BZ/(1-AZ);peak=Z.copy();integral=np.zeros_like(T)
    for dt,E,sr,dr,sz,dz,rr,rz,br,bz in segments:
        for j in range(N):
            mid=np.exp(-dr[:,j]*dt/2)*R+sr[:,j]*(-np.expm1(-dr[:,j]*dt/2))/dr[:,j]
            if p['control']=='R_clamped':mid=np.ones_like(mid)
            integral+=dt*(p['eta']*E[:,j]*mid-p['rho']*(1-E[:,j]))
            R=rr[:,j]*R+br[:,j];Z=rz[:,j]*Z+bz[:,j];peak=np.maximum(peak,Z)
    return integral/T,peak

def reference(p=P):
    E,sr,dr,sz,dz=coeff(np.array([1.]),p)
    R=sr/dr
    if p['control']=='R_clamped':R[:]=1
    return float(p['eta']*E[0]*R[0]-p['rho']*(1-E[0])),float(sz[0]/dz[0])

def periodic(T,f,p=P,method='DOP853',rtol=2e-11,absorption=False):
    on=T*f;u=1/f;k=p['kappa'];segments=[(0,on,u),(on,T,0.)]
    if absorption:
        ka=12.;M=np.array([[-ka,0],[k,-k]]);B=np.array([ka,0.])
        def step(x,v,t):
            A=expm(M*t);return A@x+np.linalg.solve(M,(A-np.eye(2))@B*v)
        A=expm(M*T);b=step(step(np.zeros(2),u,on),0,T-on)
        x0=np.linalg.solve(np.eye(2)-A,b);xs=step(x0,u,on)
        def concentration(t):
            return float(step(x0,u,t)[1]) if t<=on else float(step(xs,0,t-on)[1])
    else:
        c0=u*(-np.expm1(-k*on))*np.exp(-k*(T-on))/(-np.expm1(-k*T))
        cs=u+(c0-u)*np.exp(-k*on)
        def concentration(t):return u+(c0-u)*np.exp(-k*t) if t<=on else cs*np.exp(-k*(t-on))
    # One-cycle affine map integrated separately from the screening algorithm.
    def affine(t,y):
        E,sr,dr,sz,dz=coeff(np.array([concentration(t)]),p)
        return [-dr[0]*y[0],sr[0]-dr[0]*y[1],-dz[0]*y[2],sz[0]-dz[0]*y[3]]
    y=np.array([1.,0,1,0])
    for start,end,_ in segments:
        so=solve_ivp(affine,(start,end),y,method=method,rtol=rtol,atol=rtol/100,max_step=T/100)
        assert so.success;y=so.y[:,-1]
    R0=y[1]/(1-y[0]);Z0=y[3]/(1-y[2])
    def fun(t,y):
        E,sr,dr,sz,dz=coeff(np.array([concentration(t)]),p)
        R=1 if p['control']=='R_clamped' else y[0]
        return [sr[0]-dr[0]*y[0],sz[0]-dz[0]*y[1],p['eta']*E[0]*R-p['rho']*(1-E[0])]
    y=np.array([R0,Z0,0.]);sols=[];extrema=[]
    for start,end,_ in segments:
        so=solve_ivp(fun,(start,end),y,method=method,rtol=rtol,atol=rtol/100,max_step=T/150,dense_output=True)
        assert so.success
        points=np.linspace(start,end,1501);der=np.array([fun(t,so.sol(t))[1] for t in points])
        roots=[brentq(lambda t:fun(t,so.sol(t))[1],a,b,xtol=1e-13) for a,b,da,db in zip(points[:-1],points[1:],der[:-1],der[1:]) if da*db<0]
        extrema.extend(float(so.sol(t)[1]) for t in [start,end]+roots)
        y=so.y[:,-1];sols.append(so)
    ts=np.linspace(0,T,1201);ys=np.array([sols[0 if t<=on else 1].sol(t) for t in ts])
    C=np.array([concentration(t) for t in ts])
    trace=pd.DataFrame(dict(time=ts,u=np.where(ts<on,u,0.),C=C,R=ys[:,0],Z=ys[:,1],D_relative=np.exp(-ys[:,2])))
    return dict(T=float(T),f=float(f),J=float(y[2]/T),peak_Z=max(extrema),
                closure_residual=float(max(abs(y[0]-R0),abs(y[1]-Z0))),mean_C=float(np.trapezoid(C,ts)/T)),trace

def finite(method='DOP853',rtol=2e-10):
    rows=[];traces=[]
    for name in ['Constant','Cycle']:
        H=20.;tail=5.;T=H/12;f=.2
        switches=[(0.,20.,1.)] if name=='Constant' else [(j*T,j*T+f*T,5.) for j in range(12)]+[(j*T+f*T,(j+1)*T,0.) for j in range(12)]
        switches=sorted(switches)+[(H,H+tail,0.)]
        y=np.array([0.,1.,0.,0.,0.]);parts=[];peaks=[]
        for start,end,u in switches:
            def fun(t,y):
                E,sr,dr,sz,dz=coeff(np.array([y[0]]),P)
                return [4*(u-y[0]),sr[0]-dr[0]*y[1],sz[0]-dz[0]*y[2],.08*(1-E[0])-E[0]*y[1],y[0]]
            so=solve_ivp(fun,(start,end),y,method=method,rtol=rtol,atol=rtol/100,max_step=.02,dense_output=True)
            assert so.success
            t=np.linspace(start,end,max(10,int((end-start)*100)))
            deriv=np.array([fun(x,so.sol(x))[2] for x in t]);roots=[]
            for a,b,da,db in zip(t[:-1],t[1:],deriv[:-1],deriv[1:]):
                if da*db<0:roots.append(brentq(lambda x:fun(x,so.sol(x))[2],a,b))
            peaks.extend(float(so.sol(x)[2]) for x in [start,end]+roots)
            vals=so.sol(t);parts.append(pd.DataFrame(dict(time=t,C=vals[0],R=vals[1],Z=vals[2],D=np.exp(vals[3]),u=u,regimen=name)))
            if abs(end-H)<1e-8:stopC=float(so.y[0,-1])
            y=so.y[:,-1]
        rows.append(dict(regimen=name,administered_amount=20.,J_F=-y[3]/25,D_followup=np.exp(y[3]),peak_Z=max(peaks),
                         C_stop=stopC,C_followup=y[0],exposure_area=y[4],remaining_exposure=y[0]/4,
                         mass_balance_error=abs(y[4]-(20-y[0]/4))))
        traces.append(pd.concat(parts,ignore_index=True))
    return pd.DataFrame(rows),pd.concat(traces,ignore_index=True)

def reduced(out):
    B=1.;L=.15;D=1.;R=.2;g=.4
    rates=np.array([.5,1.,2.]);A=B*(1-R)
    def utility(w,r):return A*(-np.expm1(-r*w))-L*D*np.expm1(g*w)
    star=np.maximum(0,np.log(A*rates/(L*g*D))/(rates+g));common=minimize_scalar(lambda w:-np.mean(utility(w,rates)),bounds=(0,10),method='bounded',options={'xatol':1e-12}).x
    rows=[dict(rate=float(r),optimal_wait=float(w),individual_value=float(utility(w,r)),common_wait=float(common),common_value=float(utility(common,r))) for r,w in zip(rates,star)]
    write(pd.DataFrame(rows),out,'T08_reduced_information')
    w1=star[1];v1=utility(w1,1);zero=brentq(lambda w:utility(w,1),w1,10)
    result=dict(common_wait=float(common),common_expected_value=float(np.mean(utility(common,rates))),
                ideal_expected_value=float(np.mean(utility(star,rates))),information_gain=float(np.mean(utility(star,rates)-utility(common,rates))),
                rate1_optimal_wait=float(w1),rate1_max_value=float(v1),rate1_positive_window_end=float(zero))
    w=np.linspace(0,6,601);fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for r in rates:ax[0].plot(w,utility(w,r),label=f'r={r:g}')
    ax[0].axhline(0,color=GRAY,ls='--');ax[0].set(xlabel='Drug-free wait',ylabel='Conditional continuation value V',title='A  Reduced waiting model');ax[0].legend()
    ax[1].bar(['Common wait','Ideal rate information'],[result['common_expected_value'],result['ideal_expected_value']],color=[GRAY,BLUE])
    ax[1].set(ylim=(0,.55),ylabel='Expected V',title='B  Equal challenge and horizon')
    savefig(fig,out,'F07_reduced_restart_information')
    return result

def palbociclib(method='RK45',rtol=2e-9):
    rows=[];traces=[]
    ka=.187*24;kel=57.09/1580*24;lag=.658/24;ktr=4/5.29;base=2.92;slope=.0011;gamma=.102
    for name in ['21 ON / 7 OFF','Daily redistribution']:
        dose=125. if name.startswith('21') else 93.75
        days=[d for d in range(168) if name=='Daily redistribution' or d%28<21]
        events=[(d+lag,dose) for d in days];bounds=sorted(set([0.,168.]+[t for t,_ in events]));ev=dict(events)
        y=np.array([0.,0.,base,base,base,base,base]);nadir=base;nt=0.;maxC=0.;parts=[]
        def fun(t,y):
            gut,central,prol,t1,t2,t3,anc=y
            C=1000*central/1580
            return [-ka*gut,ka*gut-kel*central,ktr*prol*((1-slope*C)*(base/anc)**gamma-1),
                    ktr*(prol-t1),ktr*(t1-t2),ktr*(t2-t3),ktr*(t3-anc)]
        for start,end in zip(bounds[:-1],bounds[1:]):
            if start in ev:y[0]+=ev[start]
            so=solve_ivp(fun,(start,end),y,method=method,rtol=rtol,atol=1e-11 if rtol>=2e-9 else 1e-13,max_step=.05,dense_output=True)
            assert so.success and np.min(so.y[2:])>0
            m=minimize_scalar(lambda t:so.sol(t)[6],bounds=(start,end),method='bounded',options={'xatol':1e-10})
            for t in [start,end,m.x]:
                v=float(so.sol(t)[6])
                if v<nadir:nadir=v;nt=float(t)
            ts=np.linspace(start,end,max(3,int((end-start)*20)));vals=so.sol(ts)
            maxC=max(maxC,float(np.max(vals[1])*1000/1580))
            parts.append(pd.DataFrame(dict(time_days=ts,ANC=vals[6],C_ug_L=1000*vals[1]/1580,regimen=name)))
            y=so.y[:,-1]
        rows.append(dict(regimen=name,administered_mg=sum(v for _,v in events),nadir_ANC=nadir,nadir_time_days=nt,max_C_ug_L=maxC,
                         ANC_at_day168=y[6],method=method,rtol=rtol))
        traces.append(pd.concat(parts,ignore_index=True))
    return pd.DataFrame(rows),pd.concat(traces,ignore_index=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(ROOT/'results'));ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();out=Path(args.output);(out/'tables').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    if args.verify_only:
        v,tr=periodic(1.679033,.2,method='Radau',rtol=2e-12)
        f,ft=finite(method='Radau',rtol=2e-12)
        pal,pt=palbociclib(method='DOP853',rtol=2e-11)
        write(f,out,'finite_independent');write(pal,out,'palbociclib_independent')
        (out/'model_independent.json').write_text(json.dumps(v,indent=2));print(v);return
    periods=np.geomspace(.2,30,45);duties=np.linspace(.2,.95,31);T,F=np.meshgrid(periods,duties,indexing='ij');T=T.ravel();F=F.ravel()
    J,Z=screen(T,F);ref,rz=reference();grid=pd.DataFrame(dict(T=T,f=F,J_screen=J,peak_Z_screen=Z,safe_05=Z<=.5,safe_022=Z<=.22))
    write(grid,out,'S08_canonical_calendar_grid')
    opt=minimize(lambda x:-screen(np.exp(x[0]),x[1],N=1024)[0][0],[np.log(1.68),.2],method='L-BFGS-B',bounds=[(np.log(.2),np.log(30)),(.2,.95)],options={'ftol':1e-13,'gtol':1e-9})
    mainval,trace=periodic(np.exp(opt.x[0]),opt.x[1]);mainval.update(regimen='Refined cycle',delta_J=mainval['J']-ref)
    fixedval,_=periodic(1.679033,.2)
    tab=[dict(regimen='Constant',T=np.nan,f=1,J=ref,peak_Z=rz,delta_J=0.,closure_residual=0.),mainval]
    write(pd.DataFrame(tab),out,'T06_canonical_main');write(trace,out,'S09_canonical_trace')
    convergence=[]
    for n in [256,512,1024,2048,4096,8192]:
        jj,zz=screen(mainval['T'],mainval['f'],N=n)
        convergence.append(dict(cells_per_segment=n,J=jj[0],peak_Z=zz[0],abs_J_error=abs(jj[0]-mainval['J']),abs_peak_error=abs(zz[0]-mainval['peak_Z'])))
    write(pd.DataFrame(convergence),out,'T09_canonical_convergence')
    amp=[]
    for umax in [2,3,5,8]:
        oo=minimize(lambda x:-screen(np.exp(x[0]),x[1],N=1024)[0][0],[np.log(2),1/umax],method='L-BFGS-B',bounds=[(np.log(.2),np.log(30)),(1/umax,.95)],options={'ftol':1e-12})
        v,_=periodic(np.exp(oo.x[0]),oo.x[1]);v.update(umax=umax,delta_J=v['J']-ref);amp.append(v)
    write(pd.DataFrame(amp),out,'T07_canonical_amplitude')
    controls=[]
    for name in ['full','R_clamped','recovery_disabled','ordinary_repair','rebound_disabled']:
        p={**P,'control':name}
        if name=='rebound_disabled':p['rho']=0
        jj,zz=screen(T,F,p);r,z=reference(p);best=int(np.argmax(jj))
        v,_=periodic(T[best],F[best],p)
        controls.append(dict(control=name,constant_J=r,best_grid_cycle_J=v['J'],class_gain=max(0,v['J']-r),constant_Z=z,cycle_Z=v['peak_Z'],T=T[best],f=F[best]))
    write(pd.DataFrame(controls),out,'T10_mechanistic_controls')
    tightidx=int(np.argmax(np.where(Z<=.22,J,-np.inf)));tight,_=periodic(T[tightidx],F[tightidx])
    alt={**P,'gate':'exponential'};aj,az=screen(T,F,alt,N=1024);i=int(np.argmax(np.where(az<=.5,aj,-np.inf)))
    alternative,_=periodic(T[i],F[i],alt);alternative['reference_J']=reference(alt)[0]
    absorption,_=periodic(mainval['T'],mainval['f'],absorption=True)
    rob=[]
    for field,vals in [('kappa',[2,4,8]),('n',[2,4,6]),('r',[.5,1,2]),('rho',[.02,.08,.32])]:
        for value in vals:
            p={**P,field:value};jj,zz=screen(T,F,p);i=int(np.argmax(np.where(zz<=.5,jj,-np.inf)))
            v,_=periodic(T[i],F[i],p);r,z=reference(p)
            rob.append(dict(parameter=field,value=value,reference_J=r,cycle_J=v['J'],delta_J=v['J']-r,peak_Z=v['peak_Z'],T=T[i],f=F[i]))
    write(pd.DataFrame(rob),out,'T11_canonical_robustness')
    # Search-based phase map using the declared 20x22 axes and calendar grid.
    maps=[]
    for k in np.geomspace(.2,40,20):
        for rho in np.geomspace(.01,2,22):
            p={**P,'kappa':k,'rho':rho};jj,zz=screen(T,F,p,N=256);r,z=reference(p)
            for cap in [.5,.22]:
                safe=zz<=cap
                if z<=cap:
                    best=max(r,float(np.max(jj[safe])) if np.any(safe) else r);label='Objective gain' if best-r>1e-5 else 'Reference retained';delta=best-r
                elif np.any(safe):best=float(np.max(jj[safe]));label='Feasibility restoration';delta=np.nan
                else:best=np.nan;label='No sampled safe candidate';delta=np.nan
                maps.append(dict(kappa=k,rho=rho,cap=cap,label=label,J_best=best,delta_J=delta,nonpositive=bool(np.isfinite(best) and best<=0)))
    mp=pd.DataFrame(maps);write(mp,out,'S10_canonical_phase_map')
    from matplotlib.colors import ListedColormap,BoundaryNorm
    labels=['Reference retained','Objective gain','Feasibility restoration','No sampled safe candidate']
    colors=['#D8D8D8',BLUE,'#68A67D','#F1C3A6']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,cap in zip(axes,[.5,.22]):
        sub=mp[mp.cap==cap];ks=sorted(sub.kappa.unique());rs=sorted(sub.rho.unique())
        codes=np.array([[labels.index(sub[(sub.kappa==k)&(sub.rho==r)].iloc[0].label) for k in ks] for r in rs])
        ax.pcolormesh(ks,rs,codes,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4.5),4),shading='nearest')
        for _,row in sub[sub.nonpositive].iterrows():ax.plot(row.kappa,row.rho,'x',color='black',ms=4,mew=.7)
        ax.set(xscale='log',yscale='log',xlabel='Washout / recovery rate ratio κ',ylabel='Rebound / recovery rate ratio ρ',title=f'Injury cap {cap:g}')
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c) for c in colors],labels,loc='outside lower center',ncol=2,fontsize=9)
    savefig(fig,out,'F09_canonical_phase_map')
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ct=pd.DataFrame(controls)
    names=['Full','R fixed at 1','Recovery disabled','Ungated repair','No rebound']
    ax[0].barh(names,ct.class_gain,color=BLUE);ax[0].set(xlabel='Verified grid-candidate class gain in J',title='A  Mechanistic interventions')
    am=pd.DataFrame(amp);ax[1].plot(am.umax,am.delta_J,'o-',color=BLUE);ax[1].set(xlabel='Maximum administration input',ylabel='Cycle minus constant J',title='B  Amplitude-bound sensitivity')
    savefig(fig,out,'F10_canonical_controls')
    fr,ft=finite();write(fr,out,'T12_finite_startup_tail');write(ft,out,'S11_finite_trajectories')
    rd=reduced(out)
    pal,pt=palbociclib();write(pal,out,'T13_palbociclib_safety');write(pt,out,'S12_palbociclib_trajectories')
    fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for a,col,title in zip(ax.ravel(),['C','R','Z','D_relative'],['Exposure','Responsiveness','Injury','Relative disease burden']):
        a.plot(trace.time,trace[col],color=BLUE);a.axvspan(mainval['T']*mainval['f'],mainval['T'],color=ORANGE,alpha=.15)
        a.set(xlabel='Normalized time',ylabel=col,title=title)
    ax[0,0].step(trace.time,trace.u,color=GRAY,lw=.8,where='post',label='Administration');ax[0,0].legend()
    savefig(fig,out,'F05_canonical_cycle')
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    ax[0].plot(grid.T if False else periods,[grid[grid['T']==t].J_screen.max() for t in periods],color=BLUE)
    ax[0].axhline(ref,color=GRAY,ls='--',label='Constant');ax[0].axhline(0,color='black',lw=.6);ax[0].set(xscale='log',xlabel='Period',ylabel='Best screened J across duties',title='A  Restricted calendar search');ax[0].legend()
    for name,g in ft.groupby('regimen'):ax[1].plot(g.time,g.D,label=name,color=BLUE if name=='Cycle' else GRAY)
    ax[1].axvline(20,color=GRAY,ls=':');ax[1].set(yscale='log',xlabel='Normalized time',ylabel='Disease burden D',title='B  Equal amount and common tail');ax[1].legend()
    savefig(fig,out,'F06_canonical_gain_and_finite')
    fig,ax=plt.subplots(2,1,figsize=(11,6),layout='constrained',sharex=True)
    for name,g in pt.groupby('regimen'):
        color=BLUE if name.startswith('21') else GRAY
        ax[0].plot(g.time_days,g.C_ug_L,label=name,color=color,lw=.8);ax[1].plot(g.time_days,g.ANC,label=name,color=color)
    ax[0].set(ylabel='Concentration (µg/L)',title='Published-parameter simulation; no tumor endpoint');ax[0].legend()
    ax[1].axhline(1,color=ORANGE,ls='--');ax[1].set(xlabel='Time (days)',ylabel='ANC (G/L)')
    savefig(fig,out,'F08_palbociclib_matched_budget')
    result=dict(canonical_main=mainval,fixed_candidate=fixedval,constant_J=ref,constant_Z=rz,
                tight_cap_candidate=tight,alternative_gate=alternative,finite_absorption=absorption,reduced=rd,
                phase_resolution=[20,22],model_source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limits=['Canonical simulations are normalized hypotheses, not patient fits.','Grid searches and local refinements do not certify global optimality.',
                        'Phase-map safety is screened; selected trajectories are checked continuously.','Palbociclib has no tumor efficacy model.'])
    (out/'model_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
