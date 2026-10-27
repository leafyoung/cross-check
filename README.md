# cross-check — pricing library validation suite

Canonical test cases (`cases/cases.json`) run through every library runner,
diffed pairwise against the **QuantLib golden standard** and between libraries.

## Libraries & runners

| Library | Runner | Backends tested |
|---|---|---|
| quantlib (golden, pip 1.44) | `scripts/golden_quantlib.py` | — |
| quantburn (Rust/Burn) | `scripts/run_quantburn.py [--features vulkan]` | cpu, **vulkan** |
| quantcandle (Rust/Candle) | `scripts/run_quantcandle.py [--features mkl]` | cpu, mkl |
| quantorch (PyTorch) | `quantorch/.venv/bin/python scripts/run_quantorch.py` | cpu (lock pins PyPI torch — no ROCm wheel exists in it; this box has no NVIDIA GPU) |
| tf-quant-finance (JAX) | `scripts/run_tfquant.py` via `uv run` from the tf-quant repo | **gpu** (ROCM) |

## Run

```bash
.venv/bin/python scripts/golden_quantlib.py        # -> results/quantlib.json
../quantorch/.venv/bin/python scripts/run_quantorch.py   # -> results/quantorch-cpu.json
# tf-quant (from ../tf-quant-finance):
#   XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD nice -n 19 \
#     uv run python ../cross-check/scripts/run_tfquant.py
.venv/bin/python scripts/run_quantburn.py          # -> results/quantburn-cpu.json
.venv/bin/python scripts/run_quantburn.py --features vulkan
.venv/bin/python scripts/run_quantcandle.py        # -> results/quantcandle-cpu.json
.venv/bin/python scripts/run_quantcandle.py --features mkl
.venv/bin/python scripts/compare.py                # -> results/report.md
```

Env: `.venv` holds QuantLib (uv). Rust bins live in each repo (`src/bin/crosscheck.rs`),
reusing each crate's pricing modules so the tests hit the real libraries.

## Findings (after fix session, 2026-10-27)

Verdict tiers: `PASS` ≤1e-9 (lib-vs-lib) / golden tol; `PASS(f32)` ≤1e-5 (quantorch
stores model params at f32 precision — the Rust ports widened to f64); `PASS(rng)` ≤1e-2
(stochastic cases: torch RNG ≠ Xorshift64Star stream, MC noise dominates); `SKIP`
(no equivalent); anything else FAIL. Current state: **280 PASS, 0 FAIL** (crr golden
rows pass at the documented 1e-4 convention tolerance).

### RNG parity
torchquantlib cannot reproduce quantburn/quantcandle's Xorshift64Star draws — its
stochastic models use `torch.manual_seed` + `torch.randn`. The 5 stochastic cases
(asian, heston_mc, digital, barrier, lookback) are compared at MC tolerance only
(`PASS(rng)`); the observed spreads (up to 0.59 rel on the 128-path barrier)
are consistent with tiny-sample MC variance.

### Deterministic verdicts (quantorch ↔ quantburn ↔ quantcandle ↔ tf-quant)
- Bit-exact (≤1e-13): bsm×2, crr European, crr American (post-fix), bermudan,
  black76, chooser.
- f32 tier (≤3.2e-6): heston_cf, sabr_iv, sabr_price, vasicek×2, cir, hw, bk.
- Exact vs QuantLib golden: BSM, Black-76, SABR Hagan IV, Vasicek (≤1e-15),
  CIR (post harness fix), Hull-White (post fix, 8.4e-13).
- Heston analytic 1.3e-6 vs golden = integration-scheme noise (trapezoid CF vs
  QuantLib's quadrature), tolerance 1e-5.

### FIXED in the 2026-10-27 fix session
1. **tf-quant American CRR under-pricing (~5e-4).** `crr_binomial_tree.py`
   applied the early-exercise max BEFORE discounting
   (`max(continuation, intrinsic)·e^{−r·dt}`) — a weaker exercise condition than
   the standard `max(continuation·e^{−r·dt}, intrinsic)`. Fixed to
   discount-then-max; European prices unaffected. American entries in
   `crr_binomial_tree_test.py` re-baselined (old values encoded the buggy
   order; every new value verified against an independent pure-Python
   classic-CRR reference; entries where early exercise is worthless confirmed
   unchanged). tf-quant American now matches the torchquantlib family to ~1e-13;
   `black_scholes/` module 143 passed; details in tf-quant MIGRATION_STATUS.md.
2. **torchquantlib Hull-White ZCB (1.5e-3), fixed in all three repos.** The
   zero-drift closed form (no f(0,t) dependence — did not reproduce the model's
   own initial curve) was replaced with the curve-consistent Brigo-Mercurio
   (3.39) formula at t=0: P = P^M(0,T)·exp(B·(f^M(0,0) − r_t)), B=(1−e^{−aT})/a.
   Evaluated in f64 locally (the finite-difference f(0,0) cancels
   catastrophically in f32 — 0.66% low; Heston's `_tensor64` pattern), result
   cast back to the model dtype. Mirrored identically into quantburn and
   quantcandle; test literals regenerated from the updated `gen_hull_white.py`.
   hw_zcb_1y now: quantburn/quantcandle 8.4e-13 vs golden, quantorch 2.0e-8
   (f32 output quantization). Test counts: quantorch 322 passed, quantburn 121
   (cpu + vulkan), quantcandle 123 — all green.

### Black-Karasinski golden — infeasible via pip QuantLib-Python (investigated)
pip QuantLib-Python's `BlackKarasinski` exposes neither `discountBond` (BK is
not one-factor-affine, so there is no analytic form) nor tree/lattice
construction from the SWIG surface, and there is no generic model-based ZCB
engine (DiscountingBondEngine is curve-only; the Tree/Black callable-bond
engines need callable instruments and recalibrate the model to the curve at
t=0). Convention caveat for any future C++-level golden: QuantLib's BK is
curve-recalibrated by construction, so its t=0 ZCB equals the flat-curve DF
(e^{−0.03·5}=0.8607 for the case params) while torchquantlib's BK
`zero_coupon_bond_price` is a numerical integral of the lognormal dynamics
(0.7106) — a convention difference, not a bug. bk_zcb_5y stays SKIP vs golden
and is compared lib-vs-lib (quantburn == quantcandle exactly; quantorch within
the f32 tier).

### Remaining documented difference (NOT a bug; tolerance set at 1e-4)
**CRR vs QuantLib (~6e-5 at 100 steps).** Convention difference, verified by
reimplementing both trees in Python and reproducing both sides to the last
digit: the torchquantlib family (and tf-quant) use classic multiplicative CRR
(u=e^{σ√dt}, d=1/u, p=(e^{(r−q)dt}−d)/(u−d)); QuantLib's `CoxRossRubinstein`
lattice is the equal-jumps drift-centered variant (dx=σ√dt,
pu=0.5+0.5·(r−q−σ²/2)dt/dx, nodes S·e^{m·dx}). Same log-jump size, different
node placement → different O(1/steps) discretization bias. Golden tolerance
1e-4 (commented in golden_quantlib.py) still catches real bugs — the tf-quant
discount-order bug was 5e-4.

### Resolved harness bugs (not model issues)
- QuantLib's `CoxIngersollRoss` ctor is `(r0, theta, kappa, sigma)` — unlike
  Vasicek's `(r0, kappa, theta, sigma)`. With the correct order, QuantLib
  matches torchquantlib's Brigo-Mercurio CIR closed form exactly (the earlier
  "CIR closed form differs" finding was this argument-order bug).
- `BlackCalculator` receives the undiscounted forward (it discounts internally);
  double-discounting made the first black76 golden wrong.
- `ql.sabrVolatility` argument order is (strike, forward, expiry, ...).

### No QuantLib equivalent in scope (SKIP vs golden, compared lib-vs-lib)
bermudan exercise-date variant, Asian MC, BK ZCB, digital/barrier/lookback MC,
heston_mc. tf-quant scope: BSM vanilla + CRR binomial only (5 of 21 cases
priced, 16 SKIP).


## tf-quant ↔ QuantLib deep check (2026-10-28)

Expanded the catalog from 21 to 34 cases (13 new `tfq_` cases) covering the full
tf-quant ↔ QuantLib overlap:

| family | cases | tf-quant API | QuantLib golden | result |
|---|---|---|---|---|
| implied vol (Newton) | 2 | `implied_vol` | roundtrip vs true vol (market price from BlackCalculator) | 3.3e-9 / 2.3e-8 |
| implied vol (Stefanica-Radiocic approx) | 1 | `implied_vol_approx` | same, tol 1e-3 (approximation) | 5.8e-4 PASS |
| analytic Asian (discrete geometric, monthly) | 1 | `asian_option_price` | `AnalyticDiscreteGeometricAveragePriceAsianEngine` | 7.2e-12 (after aligning sampling times to whole days: QL fixing dates are ref+round(365t)) |
| analytic barrier ×4 | 4 | `vanilla_prices.barrier_price` | `AnalyticBarrierEngine` (rebate 0, continuous) | ≤1e-14 all |
| binary (cash-or-nothing) ×2 | 2 | `vanilla_prices.binary_price` | `CashOrNothingPayoff` + `AnalyticEuropeanEngine` | exact after FIX (below) |
| PDE vanilla European | 1 | `math.pde.fd_solvers.solve_backward` (docstring BS recipe + dividend drift) | `FdBlackScholesVanillaEngine` | 2.0e-4, tol 5e-3 (discretization both sides) |
| swap-curve bootstrap | 1 | `rates.swap_curve_bootstrap` (par swaps off a flat 3% continuous curve, linear-zero basis) | `PiecewiseLinearZero` + SwapRateHelpers | 3.1e-7 on DFs |
| Brownian bridge | 0 | SKIP — tff's is a barrier-hit-probability series; QL's BrownianBridge builds path variances. No shared quantity. | | |
| arithmetic Asian | 0 | SKIP — `asian_option_price` raises NotImplementedError for ARITHMETIC (asian_prices.py:201) | | |

### FIXED: `binary_price` ignored `dividend_rates`
When priced from `spots`, tff used forward `S·e^{rT}` (the accepted
`dividend_rates` argument was converted and never used) → binary calls
over-priced (ATM 1y: 0.5323 vs 0.4946 at q=2%), puts under-priced. Fixed to
`S·e^{(r−q)T}/DF·…`; verified against an independent closed-form reference
(N(d2)·e^{−rT}) reproducing both the old and new values exactly, and against
QuantLib. Existing tff binary tests use explicit `forwards` and never
exercised the spots path — unchanged, black_scholes module 143 passed.
Same `spots/DF` pattern in `implied_vol_approximation.py` /
`implied_vol_newton_root.py` is NOT a bug: those APIs take no dividend
argument (documented contract).

### Harness lessons (this round)
- QuantLib puts barrier/averaging params on the INSTRUMENT (BarrierOption,
  DiscreteAveragingAsianOption), not the engine.
- QL MC engines take no timeSteps for date-scheduled options (fixing dates drive the grid).
- swap_curve_bootstrap convention: fixed-leg cashflows are NEGATIVE (paid) —
  par swaps must pass −par; daycount fractions must be explicit (0.5 float,
  1.0 fixed) or the PV equation silently mis-accrues.
- `tff.asian_option_price` sampling_times shape is `[n] + batch_shape`.

**Final: 292 PASS / 0 FAIL / 67 SKIP** (SKIP = no QuantLib equivalent or no
tf-quant capability, each with a stated reason).
