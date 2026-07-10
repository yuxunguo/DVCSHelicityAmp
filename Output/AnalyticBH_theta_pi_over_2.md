# Fully contracted Bethe-Heitler helicity amplitudes

Planar user frame with `theta_in = pi/2`; helicities are doubled helicities.

The complete scalar formulas are in `AnalyticBH_theta_pi_over_2.tex`.
They contain no anonymous `x_i` substitutions.

## Kinematics

```text
P = (s-m^2)/(2 sqrt(s))
E = sqrt(P^2+m^2)
R = pOut, fixed by sqrt(R^2+m^2) + K + qOut = sqrt(s)
K = sqrt(R^2+qOut^2+2 R qOut sin(phiOut))
Dplus  = 2 kp.qout = 2 qOut (K+qOut+R sin(phiOut))
Dminus = -2 k.qout = -2 P qOut [1+cos(phi_in-phiOut)]
t = 2m^2 - 2 E sqrt(R^2+m^2) + 2 P R sin(phi_in)
```

## Amplitude organization

The TeX file first gives all contracted scalar leptonic numerators `N` and
proton-current components `H`, then spells out all 16 surviving amplitudes as

```text
L^nu_(h,lambda) = N^(nu,+)_(h,lambda)/Dplus + N^(nu,-)_(h,lambda)/Dminus
M_(h,s,sprime,lambda) = (1/t) sum_nu L^nu_(h,lambda) H_nu^(sprime,s)
```

The 16 channels with `hprime != h` vanish exactly for the massless electron.
