# cross-check — pricing library validation suite

Canonical test cases (`cases/cases.json`) run through every library runner,
diffed pairwise against the **QuantLib golden standard** and between libraries.

## Libraries & runners

| Library | Runner | Backends tested |
|---|---|---|
| quantlib (golden, pip 1.44) | `scripts/golden_quantlib.py` | — |
| quantburn (Rust/Burn) | `scripts/run_quantburn.py [--features vulkan]` | cpu, **vulkan** |
| quantcandle (Rust/Candle) | `scripts/run_quantcandle.py [--features mkl]` | cpu, mkl |
| quantjax (Python/JAX) | `../quantjax/.venv/bin/python -m quantjax.cli --cases cases.json` (rocm via the tf-quant uv env) | cpu, **rocm** |
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
bermudan exercise-date variant, Asian MC, BK ZCB, barrier MC,
heston_mc. (digital/lookback got exact analytic goldens on 2026-10-29 — see
the tfqm_ section; barrier stays MC-only vs golden because the family's
discrete 128-path monitoring cannot be matched to the continuous analytic.)
tf-quant scope as of 2026-10-29: 23 of 40 cases priced (BSM vanilla, CRR,
implied vol, analytic Asian/barrier/binary, PDE, swap-curve bootstrap, and
the six tfqr_ rates families).


## Coverage matrix (case × library)

Status per library: **exact** = analytic/closed-form path exercised; **MC** = Monte-Carlo
path (seeded; compared lib-vs-lib at PASS(rng) tolerance, vs golden only where a golden
exists at MC tolerance); **skip** = no equivalent (reason in findings below); **—** = not
in scope for that pairing.

| case family | quantlib (golden) | quantburn | quantcandle | quantorch | tf-quant | quantjax |
|---|---|---|---|---|---|---|
| BSM vanilla / Black-76 | exact | exact | exact | exact | exact | exact |
| CRR binomial (Euro/Amer) | exact (1e-4 convention) | exact | exact | exact | exact | exact |
| American/Bermudan tree | exact / skip¹ | exact | exact | exact | exact² | exact |
| Heston analytic | exact | exact | exact | exact | skip³ | exact |
| Heston MC | — | MC | MC | MC | — | MC |
| SABR Hagan IV / price | exact / skip | exact | exact | exact | skip⁴ | exact |
| Vasicek / CIR ZCB | exact | exact | exact | exact | skip⁴ | exact |
| Hull-White / Black-Karasinski ZCB | exact / skip⁵ | exact | exact | exact | skip⁴ | exact |
| LMM, Dupire, GAN | — | exact (parity tests only) | exact | exact | — | exact |
| Bonds (zero/coupon/callable/putable) | skip⁶ | exact | exact | exact | — | exact |
| Exotics MC (digital/barrier/lookback) | exact¹¹ (digital, lookback) / skip⁷ (barrier) | MC | MC | MC | — | MC |
| Analytic barrier (continuous) | exact | skip⁸ | skip⁸ | skip⁸ | exact | skip⁸ |
| Binary (cash-or-nothing, analytic) | exact | skip⁸ | skip⁸ | skip⁸ | exact | skip⁸ |
| Asian analytic (geometric) | exact | skip⁸ | skip⁸ | skip⁸ | exact | skip⁸ |
| Implied vol roundtrip | golden⁹ | letsberational¹⁰ | letsberational¹⁰ | letsberational¹⁰ | exact | skip |
| PDE European | exact | — | — | — | exact | — |
| Swap-curve bootstrap | exact | — | — | — | exact | — |
| Hagan-West curve / monotone convex | pure-Py ref¹² | — | — | — | exact | — |
| Nelson-Siegel-Svensson fit | true curve + QL Svensson¹² | — | — | — | exact | — |
| forwards / cashflow identities | closed form | — | — | — | exact | — |
| Greeks / VaR / ES / credit / XVA | — | MC/parity | MC/parity | MC/parity | — | MC/parity |

¹ Bermudan exercise-date variant has no QuantLib mapping in scope. ² tff's Bermudan is
SKIP (no tree with exercise dates). ³ tff has no analytic Heston. ⁴ tff has no Hagan IV,
one-factor ZCB closed forms. ⁵ BK golden infeasible via pip QuantLib-Python (SWIG exposes
no discountBond/tree; also QuantLib BK is curve-recalibrated — convention difference).
⁶ QuantLib bond engines are curve-based, different convention. ⁷ quantlib MC golden not
wired for these (quantburn/quantcandle/quantorch compare lib-vs-lib). ⁸ torchquantlib
family prices these by MC only (exotics module); analytic variants exist only in tff.
¹¹ analytic goldens added 2026-10-29 (tfqm_ section); the MC rows compare at the
1e-2 rng tolerance (INFO(rng) with deviations shown). ¹² no QL monotone-convex
builder — golden is an independent pure-Python Hagan-West reference (QL
cross-checked on the ZCB-pinned basis and for the Svensson fit; see tfqr_
section).
⁹ golden = true vol used to generate prices. ¹⁰ let's-be-rational is call-only IV (Brent/
Newton on Black) — not wired into the case catalog (different API surface).

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
tf-quant capability, each with a stated reason). After quantjax joined
(2026-10-28): 547 PASS / 0 FAIL — quantjax matches quantburn/quantcandle
bit-exactly on all 21 shared models and rocm == cpu.

## tf-quant rates (tfqr_) and MC goldens for exotics (2026-10-29)

Catalog 34 → 40 cases: six `tfqr_` rates cases (tff `rates/` vs goldens) and
two `tfqm_` analytic goldens attached to the existing `digital_call` /
`lookback_call` MC cases. `compare.py` now (a) compares dict-of-vector
payloads per key (any curve values, not just the old `dfs` shape) and (b)
no longer drops stochastic cases from the golden table: MC rows against an
exact golden are compared at the documented **1e-2 rng tolerance** with
`PASS(rng)`/`INFO(rng)` verdicts — above tolerance the row stays visible
with its deviation (same policy as lib-vs-lib MC rows; never masked).

### tfqr_ rates cases (all PASS, 0 FAIL)

| case | tff API | golden | rel err | tol |
|---|---|---|---|---|
| tfqr_hw_bondcurve | `rates.hagan_west.bond_curve` (HW 2008 paper's 4-bond example) | independent pure-Python HW reference in `golden_quantlib.py` | 4.0e-11 | 1e-6 |
| tfqr_hw_monotone | `hagan_west.monotone_convex.interpolate_yields/interpolate_forward_rate` (incl. negative-forward curve) | same reference | 0.0 (exact) | 1e-9 |
| tfqr_hw_zcb | `bond_curve` on a 6-point ZCB strip (day-aligned d/365 maturities, NSS true curve) | true DFs + QL `PiecewiseLogLinearDiscount` deposit bootstrap cross-check (1.2e-13) | 2.4e-16 | 1e-9 |
| tfqr_nss_fit | NSS fit = harness-side L-BFGS (tff `math.optimizer.bfgs_minimize`) over tff `nelson_seigel_svensson.interpolate`, log-tau parametrization | true NSS curve at test points + QL `SvenssonFitting`/`FittedBondDiscountCurve` recovery cross-check (6e-13) | 2.6e-12 | 1e-9 |
| tfqr_forwards | `rates.analytics.forwards`: forward_rates_from_yields / yields_from_forward_rates / forward_rates(DFs) | closed-form pure Python | 3.2e-15 | 1e-9 |
| tfqr_cashflows | `rates.analytics.cashflows`: pv_from_yields / yields_from_pv / present_value | closed-form annuity + input yield | 1.2e-16 | 1e-8 |

Tolerance justifications (also commented in `golden_quantlib.py`):
- `tfqr_hw_bondcurve` 1e-6: both sides stop on the same 1e-8-discount-factor
  iteration criterion from slightly different iterates; measured 4e-11,
  headroom for the O(1e-8) stopping residual. Both curves re-price the input
  bonds to <1e-9.
- `tfqr_hw_monotone` 1e-9: identical closed-form scheme, independent code
  paths; measured agreement exact.
- `tfqr_hw_zcb` / `tfqr_nss_fit` / `tfqr_forwards` / `tfqr_cashflows` 1e-9
  (cashflows 1e-8 for the Newton yield tolerance): exact identities or
  machine-precision recovery on pinned quantities; measured 1e-16..1e-12.

### QL-vs-Hagan-West scheme difference (documented, not a bug)
QuantLib's pip surface has no monotone-convex builder; its
`PiecewiseLogLinearDiscount` bootstraps log-linear discounts while Hagan-West
interpolates instantaneous **forwards**. For coupon instruments the two bases
give different node discount factors (the intermediate coupon DFs depend on
the scheme), so golden-vs-QL is not meaningful for the HW bond bootstrap —
hence the independent pure-Python reference (written from the Hagan-West
2006/2008 papers; verified by interval-integral exactness 1e-17, endpoint
continuity 5e-17, the paper's published bootstrap table to 9e-14, bond
re-pricing 3e-12, and a single bootstrap step matching tff digit-for-digit).
On the shared basis where both schemes are pinned — a zero-coupon strip, node
DF = price/face — QL and tff agree to 2.4e-16 (`tfqr_hw_zcb`).

### NSS notes
tff ships the NSS **interpolation** only; the fit is harness-side (tff's own
BFGS + interpolate, taus in log space to keep them positive). The synthetic
data lies exactly on the model manifold and the compared test points are fit
tenors, so the target is independent of the (non-unique) fitted parameters —
tff recovers the true curve to 2.6e-12 and even the parameters themselves to
~1e-6. QL's `SvenssonFitting` on ZCBs priced off the true curve recovers the
curve to 6e-13 (asserted inside the golden; a mismatch raises as a FAIL).

### tfqm_ MC goldens (digital_call / lookback_call)
- **digital_call**: golden = QL analytic `CashOrNothingPayoff` (payout 10)
  = 5.3232482 — exact analytic preferred over MC. Verified against the
  pure-Python closed form `payout·e^{−rT}·N(d2)` (2e-15) and a 500k-sample
  terminal-GBM MC (5.3258 ± 0.0067).
- **lookback_call**: pip QuantLib 1.44 **does** expose
  `AnalyticContinuousFixedLookbackEngine` (Conze-Viswanathan), so no SKIP was
  needed. Golden = 19.1676253 with CONTINUOUS monitoring; verified against
  the pure-Python CV formula (2e-14) and a 500k-sample 252-step MC:
  18.375 ± 0.022.
- **Convention finding (documented, not masked):** the torchquantlib-family
  lookback MC monitors the running max/mini at 252 discrete steps/year, so
  its expectation sits ~4.1% BELOW the continuous-monitoring golden —
  quantitatively the Broadie-Glasserman-Kou discrete-monitoring adjustment
  (e^{−0.5826·σ·√dt} ≈ −0.73% on the max, ≈ 0.82 on this price). All MC rows
  (Xorshift64Star vs the analytic golden) land at `INFO(rng)` at the 1e-2
  tolerance: digital 8.9% (burn family) / 3.0% (orch) against a 256-path
  rel SE of ~5.6%; lookback 7.9% / 7.2% against a 128-path rel SE of ~7%.
  Deviations are printed in the report rows; they are MC-noise-plus-monitoring-
  convention, consistent with the 500k MC cross-checks above.

### Final tallies (2026-10-29)
vs golden: **107 PASS, 14 INFO(rng), 93 SKIP, 0 FAIL**. Lib-vs-lib: **323
PASS, 75 PASS(rng), 30 INFO(rng), 48 PASS(f32), 0 FAIL**. SKIPs each carry a
reason (no QL equivalent / no capability in that library). tff now prices 23
of 40 cases; the remaining tff SKIPs are models tff does not have (Heston
analytic, SABR Hagan, one-factor ZCB closed forms, MC exotics, tree
Bermudan, chooser, bonds, BK).
