# Equations and computation

## Canonical model

The normalized state is exposure C, responsiveness R, injury Z and positive burden D. Integrators evolve log D or its negative, to avoid underflow. Time is in responsiveness-recovery units; input u is concentration-equivalent, not a clinical mass rate.

```text
dC/dt = κ(u − C)
dR/dt = r W(C)(1 − R) − α E(C)R
dZ/dt = β E(C) − π WZ(C)Z
d log D/dt = ρ(1 − E(C)) − η E(C)R
E(C)  = C/(1 + C)
W(C)  = 1/(1 + (C/χ)^n)
WZ(C) = 1/(1 + (C/χZ)^2)
```

Defaults are κ=4, α=2, π=0.7, β=0.2, η=1, ρ=0.08, χ=0.35, χZ=1, n=4, r=1. The multiplier r changes recovery only in the parameter-box experiment. In a common physical conversion, u=Fbio Uadm/(CL KE); equal mean u implies equal mean administered mass only when these factors are common.

Constant input u=1 is compared with a rectangular cycle: ON input 1/f on [0,fT), then zero on [fT,T). Mean input is one. The principal calendar screen uses 45 logarithmic periods on [0.2,30] and 31 duties on [0.2,0.95] (1,395 rectangles). Refinement uses 1,024 cells per segment; the main duty bound caps peak input at five. Periodic C is analytic. Independent ODE integration constructs affine maps for periodic R,Z. D is not periodic.

```text
J = η mean(E R) − ρ mean(1 − E)
```

J is the long-run log-decline rate, with positive J indicating decline within this model. Injury must obey Z≤0.5, with Z≤0.22 as a tighter comparison. Dense-path checks evaluate endpoints, switches and bracketed roots of dZ/dt. Screens approximate the trajectory; they do not certify safety for every calendar.

The finite equal-amount comparison starts C=0,R=1,Z=0,D=1. Both administer 20 normalized units through time 20; follow-up ends at 25. The cycle has 12 periods of 20/12, duty 0.2 and ON input five. The finite endpoint is −log D(25)/25, not periodic J. Exposure mass balance includes remaining exposure C(25)/κ.

Whole-family comparisons distinguish a shared available cap from equal realized administration. Stationary constants have u∈[0,1], injury cap 0.22, peak cap five and mean-input cap one. The finite constant-family numerical upper bound relaxes injury and uses a 4,096-interval input grid, an analytic derivative bound and a 1e−7 integration allowance. These are conservative numerical bounds, not formal interval-arithmetic proofs.

Four separately named candidate traces retain distinct periods: A-PER-REFINED 1.679033; A-PER-RERUN 1.679027411422; A-AMP-5 1.679036617960; A-CAP-EXT 1.679903. Equal rounded objectives do not make the controls identical.

Mechanism controls clamp R=1, disable responsiveness restoration, remove repair gating or set rebound ρ=0. The alternative gate is exponential. A first-order absorption scenario uses normalized ka=12. A phase map screens 20 κ values by 22 ρ values at two injury caps. The extension uses 32 deterministic Sobol points spanning ±20% of nine parameters; those points are not a patient distribution.

## Oral PK and neutropenia

Palbociclib parameters: ka=0.187 h⁻¹, CL/F=57.09 L/h, V/F=1,580 L, lag=0.658 h, baseline ANC=2.92 G/L, effect slope=0.0011 L/µg, MTT=5.29 days and feedback exponent γ=0.102. The implementation converts hourly rates to daily rates and uses ktr=4/MTT.

```text
dGut/dt = −ka Gut; lagged dose events increment Gut in mg
dCentral/dt = ka Gut − kel Central; kel=(CL/F)/(V/F)
C = 1000 Central/(V/F)                  # µg/L
dProl/dt = ktr Prol [(1 − slope C)(baseline/ANC)^γ − 1]
dT1/dt = ktr(Prol − T1)
dT2/dt = ktr(T1 − T2)
dT3/dt = ktr(T2 − T3)
dANC/dt = ktr(T3 − ANC)
```

All five PD pools start at baseline; PK compartments start empty. Apparent clearance and volume require no extra bioavailability multiplier. Six 28-day cycles of 125 mg on days 0–20 are compared with daily 93.75 mg for 168 days. Both administer 15,750 mg. Event-split RK45 and DOP853 simulations evaluate ANC extrema and extend through day 182. An eliminated-mass state verifies gut+central+eliminated=administered. The endpoint is minimum ANC; no tumor efficacy or patient variability is modeled.

Parameter source: Le Marouille et al., *Pharmacokinetic/Pharmacodynamic Model of Neutropenia in Real-Life Palbociclib-Treated Patients*, Pharmaceutics 2021;13:1708, [doi:10.3390/pharmaceutics13101708](https://doi.org/10.3390/pharmaceutics13101708).

## Conditional waiting model

```text
R(w) = 1 − (1 − R0) exp(−rw)
D(w) = D0 exp(gw)
V(w) = B(1 − R0)[1 − exp(−rw)] − L D0[exp(gw) − 1]
w* = max(0, log[B r(1 − R0)/(L g D0)]/(r + g))
```

Use B=1,R0=0.2,L=0.15,D0=1,g=0.4 and equally probable rates r=0.5,1,2. The ideal example shares wait bounds [0,10]. The noisy-information example shares [0,6], uses Y=log(r)+Normal(0,σ²), and integrates posterior-optimal decisions with 80 and 160 Gauss–Hermite nodes for σ=0,0.25,0.5,1,2. Monitoring costs 0.002 and 0.01 are hypothetical. V is a separate conditional model, not the canonical ODE's value function or a fitted clinical utility.
