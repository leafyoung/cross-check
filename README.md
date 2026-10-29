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
.venv/bin/python scripts/risk_golden.py            # -> results/risk_golden.json (needs ../quantjax and ../quantorch venvs)
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

### Black-Karasinski golden — RESOLVED 2026-10-29 (was "infeasible via pip QuantLib-Python")

The earlier investigation missed a route: pip QuantLib 1.44 DOES expose
`CallableZeroCouponBond` + `TreeCallableFixedRateBondEngine`, which together price a
plain zero-coupon bond (empty callability schedule) off the `ql.BlackKarasinski`
short-rate tree. BK itself still exposes no `discountBond` (not one-factor-affine) and
`DiscountingBondEngine` is curve-only — but the tree-engine route is a valid golden.
bk_zcb_5y is now golden-priced under the QUANTLIB convention; the torchquantlib family
rows render CONVENTION against it. Full route, tree-convergence measurement and the
two-convention analysis: "Black-Karasinski golden — two conventions, both verified"
section at the end.

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
bermudan exercise-date variant, Asian MC, barrier MC,
heston_mc. (digital/lookback got exact analytic goldens on 2026-10-29 — see
the tfqm_ section; barrier stays MC-only vs golden because the family's
discrete 128-path monitoring cannot be matched to the continuous analytic.
BK ZCB got a golden on 2026-10-29 — see the BK section.)
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
| Hull-White / Black-Karasinski ZCB | exact / exact⁵ (convention) | exact | exact | exact | skip⁴ | exact |
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
| Greeks / VaR / ES / credit / XVA | golden¹³ | MC/parity | MC/parity | MC/parity | — | MC/parity |

¹ Bermudan exercise-date variant has no QuantLib mapping in scope. ² tff's Bermudan is
SKIP (no tree with exercise dates). ³ tff has no analytic Heston. ⁴ tff has no Hagan IV,
one-factor ZCB closed forms. ⁵ BK golden = `CallableZeroCouponBond` (empty
callability) priced by `TreeCallableFixedRateBondEngine` off
`ql.BlackKarasinski`; the QL BK is curve-recalibrated, so the golden =
flat-curve DF up to tree error (measured 5.2e-16 at 400 steps, tol 1e-12).
The torchquantlib family computes a different convention
(expected-lognormal-path integral from r0, 0.7106 vs 0.8607) — flagged
`convention: true` in cases.json, rows render CONVENTION (see BK section).
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
¹³ risk_golden (2026-10-29): Merton exact, XVA SKIP (no QL engine), VaR/ES
convention-checked vs RiskStatistics, family Malliavin greeks FAIL(bias) — see
the risk-modules findings section below.

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


## 9-item improvement round (2026-10-29)

**Item 1 — tf-quant rates vs goldens (6 tfqr_ cases, all PASS).** Hagan-West
verified against an independent pure-Python reference written from the papers
(QL pip has no monotone-convex builder — scheme difference documented; QL
compared only on the shared ZCB-pinned basis, 2.4e-16). NSS fit (tff
interpolation + harness BFGS) recovers the true curve to 2.6e-12, QL
SvenssonFitting cross-assert 6e-13. forwards/analytics identities ≤3.2e-15.

**Item 2 — exotic goldens.** digital_call: QL analytic CashOrNothing (payout
10) = 5.3232482. lookback_call: QL AnalyticContinuousFixedLookbackEngine
(Conze-Viswanathan) = 19.1676253. MC rows land at INFO(rng) — the burn-family
lookback sits 7.9% low, quantitatively the Broadie-Glasserman-Kou discrete-
monitoring bias e^(−0.5826·σ·√dt) (252 steps/yr) — convention, documented.

**Item 3 — analytic exotics for the whole family.** tff's analytic barrier +
binary ported into quantburn/quantcandle/quantorch/quantjax
(vanilla_analytics; verified vs QL to 4.6e-14 / 3.3e-16 and vs tff f64 to
1.1e-14 incl. rebates; in+out=vanilla parity 3.6e-15). One documented
convention: tff pays KO rebates at hit, QL at expiry (~3e-2 on rebates).

**Item 4 — scripts/run_all.sh.** One-command full pipeline (all runners +
compare), per-step failure capture. Fixed: absolute paths for the quantjax
steps; compare.py now skips non-runner JSONs (calibration.json).

**Item 5 — scripts/invariants.py.** I1 put-call parity, I2 AoN−K·CoN=vanilla,
I3 barrier in+out=vanilla, I4 CRR(500)→BSM (3.8e-4, tree convention), I5 ZCB
monotonicity — checked per library where the quantity exists: **0 violations**
(results/invariants.md).

**Item 6 — calibration cross-check (scripts/calibration_check.py).** Heston
surface generated by QL AnalyticHestonEngine from known params: quantorch
recovers to ≤9.7e-4 rel (Adam/float32), QL LM to ~1e-6; engines agree ≤9.8e-4.
SABR recovery: quantorch 3e-7, reference Nelder-Mead 1e-13. No local minima,
no constraint clamping. Follow-ups noted (noisy quotes, multistart, float64
params).

**Item 7 — tf-quant flaky triage (tff commit a9374ea8).** Suite 12 failed /
1344 passed → **6 failed / 1347 passed**. Four real porting bugs fixed (CG
batch backtracking jnp.all→any; optimizer stop-condition flag semantics;
top_k tie-break in adaptive_update; TF-2.21 legacy Philox seeding
reverse-engineered and wired — bit-exact vs TF reference streams). Parked:
swaption negative-tau, hagan_west unstable (fails under real TF too), svi
outlier test, f32 iteration count.

**Item 8 — baseline tags.** All six repos tagged
cross-check-baseline-2026-10-28 (no remotes exist — local only).

**Item 9 — tf-quant housekeeping (commit caca2a17).** Landed the gfx1151
memory-defense work (sitecustomize, conftest, AGENTS.md consolidation,
removed the completed llm-refactor-plan.md).

**Final: 553 PASS / 0 FAIL / 135 SKIP** (40 cases × 7 runners).

## Risk modules vs QuantLib golden (2026-10-29)

First independent verification of the torchquantlib family's risk modules
(`scripts/risk_golden.py` -> `results/risk_golden.json`, 46 rows). These modules
were previously only parity-tested family-internally, against reference
generators sharing the same formulas and the same Xorshift64Star stream —
near-tautological. Family channel: quantjax (bit-exact vs quantburn/quantcandle
per the parity suite); greeks called with the documented family test seed 3001
at 20k paths, plus a 1M-path convergence probe (seed 7). Goldens: QuantLib 1.44
`BlackCalculator` / `RiskStatistics` + pure-Python closed forms, cross-asserted
inside the script (a golden-source mismatch raises); the family's quantile
convention is additionally pinned against upstream `torch.quantile` (quantorch
venv, float64).

### European/digital Malliavin greeks: FAIL(bias) — real estimator bug in all family repos

| quantity | family @20k/3001 | QL analytic | rel gap | family 1M limit |
|---|---|---|---|---|
| delta | 93.9733 | 0.636831 | 1.5e+2 | 91.3232 (143x) |
| gamma | 7.26868 | 0.0187620 | 3.9e+2 | 6.88057 (367x) |
| vega | 7991.19 | 37.5240 | 2.1e+2 | 7746.22 (206x) |
| theta | −230.131 | −6.41403 | 3.5e+1 | −224.229 (35x) |
| rho | 1395.59 | 53.2325 | 2.5e+1 | 1375.66 (26x) |
| digital delta | −0.142140 (sign flip) | 0.187620 | 1.8e+0 | −0.139705 |
| digital gamma | 0.00294877 (sign flip) | −0.00328335 | 1.9e+0 | 0.00290367 |
| digital vega | −33.2999 | −6.56671 | 4.1e+0 | −32.662 |

Diagnosis: the original torchquantlib `MalliavinGreeks`
(`quantorch/torchquantlib/core/risk/greeks/greeks.py`, ported verbatim into
quantburn `src/greeks.rs`, quantcandle and quantjax) multiplies the PAYOFF by
pathwise-derivative quantities of S_T (`Dt_ST = S_T·Z/√T`, `D_sigma_ST =
S_T·(Z√T−σT)`, ...) where the Malliavin integration-by-parts weights require
the duality step (move the derivative onto the weight). Example: the correct
single-step GBM delta weight is `Z/(σ·S0·√T)` (the likelihood-ratio weight);
the family uses `S_T·Z/(σ·S0·√T)` — an extra multiplicative S_T — and its
vega/rho use the payoff where the payoff DERIVATIVE is required.

Evidence that this is bias, not MC noise (all in the script's evidence rows):
- the 1M-path family limits stay at 26x..367x the analytic value;
- an independent correct-weight estimator (own derivation, own RNG, same
  estimator class and path counts) converges to the QL analytic: 1e-3..2e-3
  rel at 1M paths; its measured 20k-path noise floor is 1.3e-2 (vega) to
  8.8e-2 (gamma) — the 1e-2 PASS(rng) screen is consistent with that floor,
  and every family row exceeds it by 1-2 orders of magnitude;
- for the digital delta, the independent pathwise-kernel estimator with the
  family's own ε=2.0 kernel is within 0.4% of analytic (ε sweep 2.0/0.5/0.1
  converges to analytic as ε→0) — the wide Gaussian-Dirac kernel is benign;
  the weight is the bug.

NOT fixed here (family repos are read-only in this workspace) — upstream
finding for torchquantlib and all four ports. The internal 1e-6 parity tests
pass because the reference generators encode the same formulas.

### VaR / ES: PASS(exact) vs spec; CONVENTION vs QuantLib

- vs the independent re-derivation of the documented semantics (losses =
  max(−r,0); torch.quantile "linear"/type-7 quantile; ES = mean{losses≥VaR} +
  1e-6·VaR): exact (0.0 rel) on both the 10-point family test sample and an
  independent 250-point LCG sample, at p ∈ {0.90, 0.95, 0.99}; cross-asserted
  against upstream `torch.quantile` (float64) to 1e-12. The family implements
  exactly what it documents.
- vs QL `RiskStatistics` fed the SAME samples (gains convention): pip
  RiskStatistics exposes NO interpolation-type argument. The conventions were
  established empirically and cross-asserted against the library to 1e-12:
  QL `valueAtRisk(p)` = −min(sorted_gains[ceil(n·(1−p))−1], 0) — a ceil order
  statistic with no interpolation; the family uses a linearly interpolated
  (type-7) quantile of losses. QL `expectedShortfall(p)` = mean of gains
  strictly below that order statistic (raises when the tail is empty — it does
  on the 10-point sample; the family ES is reported unmasked); the family uses
  mean{losses≥VaR} + 1e-6·VaR. Measured deviations on the 250-point sample:
  VaR ≤ 2.5e-2 rel, ES ≤ 1.8e-2 rel — a quantified CONVENTION difference
  (order statistic vs interpolation, strict-below vs ≥ tail), not a bug.

### Merton structural model: PASS(exact)

4 cases vs QL `BlackCalculator` + the closed form (golden sources cross-asserted
to 1e-12): worst 7.0e-16 rel at tolerance 1e-10 (the family ships its own erfc
implementation, hence not 1e-12).

### XVA (CVA/DVA/FVA/MVA): SKIP(ql), PASS(exact) vs independent re-derivation

No QuantLib engine maps to these single-period undiscounted expected-loss
products (`MidPointCdsEngine` prices CDS spreads off a hazard curve — a
different quantity). Golden = independent re-derivation of the documented
closed forms: exact (1e-12) on all four.

### Tallies

46 rows: 20 PASS(exact), 1 PASS(rng), 5 INFO(rng) (correct-estimator noise-floor
evidence at 20k), 12 CONVENTION (VaR/ES vs QL), 8 FAIL(bias) (family Malliavin
greeks, European + digital). Widest gap: European gamma, 3.9e+2 rel at 20k
paths (367x at the 1M limit) — the largest discrepancy found in this repo so
far, and the first real bug in the torchquantlib family (all prior case-level
pricing matched goldens).


## Black-Karasinski golden — two conventions, both verified (2026-10-29)

bk_zcb_5y had been golden-SKIP under the belief that the pip SWIG surface exposes no
BK tree pricing. Wrong: the missing piece was never the model, it was the instrument —
`ql.BlackKarasinski` alone exposes no `discountBond` (BK is not one-factor-affine), but
pip 1.44 DOES expose `CallableZeroCouponBond` and `TreeCallableFixedRateBondEngine`,
and a `CallableZeroCouponBond` with the default EMPTY callability schedule is a plain
zero-coupon bond. Each step below was verified with a tiny probe script before wiring.

Route (in `golden_quantlib.py::golden_black_karasinski`):
- `ql.BlackKarasinski(FlatForward(3%) handle, a, sigma)` — QuantLib recalibrates the
  model to the term structure (the theta/x0 path is fit to the curve); the supplied
  a and sigma are kept.
- `ql.CallableZeroCouponBond(0, 100.0, NullCalendar, maturity_date, Act/365F)`,
  no callability dates, redemption 100.
- `ql.TreeCallableFixedRateBondEngine(bk, 400)` — bond NPV/100 = ZCB off the BK
  trinomial tree. Evidence the price comes from the tree, not a curve discount: the
  finite-step NPV depends on a/sigma (a=0.5, 50 steps: 6.6e-11 below the curve DF;
  a=0.1, sigma=0.05: 6e-11 above) and converges to the curve DF as steps grow.

### Tree convergence (case params a=0.1, sigma=0.01, r0=0.03, T=5y; curve DF e^{-0.03*5} = 0.8607079764250578)

| steps | 25 | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 | 6400 |
|---|---|---|---|---|---|---|---|---|---|
| rel err vs DF | 8.5e-13 | 9.6e-14 | 1.1e-14 | 2.6e-15 | 5.2e-16 | 1.3e-15 | 1.6e-12 | 6.7e-13 | 5.8e-13 |

Non-monotone: the discretization error sinks to machine noise around 200-800 steps and
roundoff in the taller trees takes over again. Golden uses 400 steps (measured
5.2e-16); golden tolerance 1e-12 = measured x ~2000 margin, still far below any real
bug scale. The golden cross-asserts (a) tree NPV/face within 1e-10 of the curve DF and
(b) the family reference below reproduces the family value — a mismatch raises and
surfaces as a golden failure, never masked.

### What each convention computes

- **torchquantlib family** (quantburn/quantcandle/quantjax f64; quantorch f32):
  P = exp(-∫₀ᵀ E[r_t]dt) where r_t follows the lognormal (log-OU) dynamics started at
  r_t = r0, E[r_t] = exp(ln(r0)e^{-at} + σ²(1-e^{-2at})/(2a)), evaluated as a 100-point
  trapezoid with NO term-structure recalibration. Value for the case params:
  0.7105591541535006 — reproduced bit-for-bit by an independent pure-Python port of the
  quantburn `zero_coupon_bond_price` algorithm (`_bk_family_zcb` in golden_quantlib.py;
  pure Python floats, no library). quantorch's 0.7105591893196106 is the same n=100
  integral in f32 (4.9e-8 rel — the f32 tier). Note this model's implied 5y ZCB yield
  is ~6.9%, not the 3% input curve: the family never recalibrates to any curve.
- **QuantLib**: the model is by construction recalibrated to the initial term
  structure, so its t=0 ZCB equals the curve discount factor 0.860708 up to tree error
  (the market-standard curve-consistent BK convention).

The 1.74e-01 relative gap between 0.7106 and 0.8607 is this convention difference,
NOT a bug — neither number is wrong for what its library computes.

### Wiring and verdict rendering (2026-10-29 run)

cases.json marks bk_zcb_5y with `"convention": true` + `"convention_reason"` (the
documented reason, shown verbatim in the report row); `compare.py` renders any
over-tolerance row of a flagged case as **CONVENTION (reason)** in the vs-golden table
and counts it separately in the summary — visible, never FAIL, never silently dropped.
Result: quantburn-cpu/vulkan, quantcandle-cpu/mkl, quantjax-cpu/rocm,
quantorch-cpu all render CONVENTION (price 0.7105591/0.7105592 vs golden
0.8607079764, rel 1.74e-01, tol 1e-12; the quantorch row keeps the documented f32
widening); tfquant-gpu stays SKIP (tff has no BK); lib-vs-lib rows unchanged
(f64 family 0.0 PASS, quantorch PASS(f32) 4.95e-08). Report tallies this round:
vs golden **107 PASS / 7 CONVENTION / 14 INFO(rng) / 136 SKIP / 0 FAIL**; lib-vs-lib
**446 PASS / 75 PASS(rng) / 48 PASS(f32) / 30 INFO(rng) / 0 FAIL**. (The row counts
grew vs the morning tally because the quantburn-vulkan and quantcandle-mkl result
files were added since the previous compare run; compare.py also now skips
risk_golden.json — its rows are 'case'-keyed — instead of crashing on it.)


## Close-out round (2026-10-30)

- **Item 7 — BK golden SOLVED** (`295bffa`): pip QuantLib's
  `BlackKarasinski` + `CallableZeroCouponBond` + `TreeCallableFixedRateBondEngine`
  prices a ZCB off the BK trinomial tree (the earlier "infeasible" finding missed
  the instrument, not the model). Tree converges to the curve DF at 400 steps
  (5.2e-16); golden tolerance 1e-12 documented. New compare.py verdict:
  **CONVENTION** (7 rows) — the family's BK integral (exp(−∫E[r]dt) from r0,
  0.7106) vs QuantLib's curve-recalibrated convention (= curve DF, 0.8607) are
  two different quantities under one name; both verified against independent
  references.
- **Item 6 — KO-rebate flag, premise CORRECTED** (all four repos): QuantLib's
  AnalyticBarrierEngine pays KO rebates at HIT (proven by R vs R=0 probes +
  400k-path Brownian-bridge MC) — the earlier "timing convention" was actually
  tff's nonstandard b=(μ²+2σ²r)/σ² parameter vs the standard λ. The new
  `rebate_paid_at_expiry` flag (default False = unchanged) implements the
  mathematically correct expiry convention, QL-verified literals embedded.
  BONUS pre-existing bug found: family DI-put K>B mask row applies a hit-style
  rebate weight to a knock-in (documented, untouched).
- **Item 5 — BGK continuous lookback** (all four repos): Broadie-Glasserman-Kou
  correction (β=0.5826, max·e^{+βσ√dt}) lands within 0.008–0.36% of the
  continuous golden vs −4.4–4.6% discrete. The price-level gap is the extremum
  factor leveraged ~6x through moneyness.
- **Item 4 — drift-centered lattice mode** (all four repos): optional
  `lattice="drift_centered"` reproduces QuantLib's CRR literals BIT-EXACTLY
  (Euro 10.429985954340486, Amer 6.0826182179030805); "crr" default unchanged.
- **Item 3 — float64 calibration**: quantorch gains dtype plumbing
  (set_dtype/to, default float32 unchanged). Float64 Heston recovery: every
  param 2.2–3.2x better at matched wall time (+4.5%); with a fine lr tail it
  reaches QL-LM-level recovery on sigma_v (5.9e-7).
- **Item 2 — andersen_lake**: the OOM was mis-attributed; real cause is
  unbounded interval doubling in gauss_kronrod's adaptive loop. Dormant
  `max_intervals=65536` guard added (bit-identical on convergent paths);
  documented in tff MIGRATION_STATUS (`54a15480`).
- **Item 1 — swaption "negative-tau" DISPOSITION: misdiagnosed.** The parked
  note was wrong: upstream's 0.22811866 is genuine — an independent pure-Python
  reference of the upstream analytic path reproduces ALL FOUR expected values
  exactly (the upstream `maturities < expiries → 0` branch is upstream TF, not
  a migration artifact). Our analytic path already matched. The simulation
  variant failed from THREE stacked porting bugs in the scan rewrite of
  `_sample_paths` (dead keep_mask reads → NameError; swapped scan
  carry/index args; boolean mask used as gather indices — the shim's gather
  silently returns everything). Fixed (`bdb28c21`): swaption_test 12/12;
  full tff suite **4 failed / 1349 passed** (from 6/1347; remaining four are
  the documented semantic parks).
- **Greeks bias FIXED** (all four repos): European delta/gamma/vega/theta/rho
  now use verified LR/pathwise estimators (QL-verified at 500k paths: delta
  ≤1.4e-3, gamma ≤1e-2); digital gamma weight corrected; barrier/lookback/
  basket/asian marked BIASED/UNDER REVIEW with the recipe documented.
- **Item 8/9** — baseline tags + housekeeping commits landed earlier this
  round; all 6 repos re-bundled would capture the new state (backups at
  ../pricing-backups are from the pre-round state).
- **run_all.sh** — calibration_check is now phase-based and wired as four
  steps; full pipeline: **553 PASS / 0 FAIL / 136 SKIP / 7 CONVENTION**.
