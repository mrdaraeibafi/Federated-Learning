# Quadratic SGD + marked CTMC

## Run

```bash
pip install numpy scipy matplotlib
python -m unittest test_quadratic_ctmc_mse.py
python quadratic_ctmc_mse.py --config quadratic_config.json --out quadratic_results
```

Keep `limited_ctmc_kbatch_sync_lumped_grouped_metrics.py` alongside the script.
The existing CTMC module is not modified. The old CSV interface is not reused:
these results are deliberately separate from the previous SGD pipeline.

Edit explicit `configurations` (K,B,J) and worker `groups` (count, rate).
b=B*J is computed, not supplied. m is the mini-batch sample count, not b.
Example settings are illustrative, not calibrated paper experiments.

## Noise and learning assumptions

F(theta)=a/2*(theta-theta_star)^2. MSE means expected squared parameter error,
not prediction MSE. sigma2 is per-sample variance. Each worker result has
variance sigma2/m; a mean of K independent results has variance sigma2/(m*K).
The simulation generates aggregate Gaussian noise directly using the square
root of this LAST variance. No second division by K occurs.

One global learner, shared optimum, independent homogeneous additive noise,
no staleness; timing is independent of learning. Worker completions are tokens,
not gradients computed at old iterates. J does not introduce an extra time factor.
Dutta is specialized to c=L=a and M_G=0; eta<=1/(2a) is enforced.

## Exact time calculation

H[i,j] sums aggregation event rates INCLUDING i=j. With Q the original state
generator and initial row distribution alpha:

G(z,t) = alpha exp((Q+(z-1)H)t) 1.

q=1-eta*a; s0=(theta0-theta_star)^2;
f=eta^2*sigma2/(m*K*(1-q^2)); d=eta*sigma2/(a*m*K).

Exact round MSE: f+(s0-f)*q^(2r).
Dutta round bound: d+(s0-d)*q^r.
Exact time MSE: f+(s0-f)*G(q^2,t).
Dutta time bound: d+(s0-d)*G(q,t).

The time bound follows by conditioning on the independent aggregation count.
The rate approximation replaces G(q^2,t) by q^(2*A*t); it is NOT an exact
fixed-time result. It is always exported to CSV, plotted only when requested.
Stationary initialization starts learning at theta0 in a stationary SYSTEM
state (not stationary learning). `initial_system: empty` is also supported.

Simulation samples physical CTMC completion channels, including self-events.
It is a learning/numerical check, not the independent worker DES used previously.
Between updates parameters are held constant. Round and time SGD samples use
separate noise streams. Shading is a pointwise normal-approximation 95% CI of
the mean across independent replications, not a simultaneous confidence band.
Dutta is a dashed theoretical upper bound, not a statistical confidence band.

## Outputs and limits

PNG/PDF round and time figures; per-configuration CSV; summary.json with full
settings, rates, floors and checks. No old files are overwritten outside the
specified output directory. Duplicate configurations get distinct numbered CSVs.

This implementation uses the existing dense, enumerated CTMC; large B,J or
worker populations may be expensive. Gaussian simulation means are noisy;
their CI may cross a theoretical bound. Increase replications for precision.
Zero-noise experiments are valid but zero MSE cannot appear on a logarithmic axis.
