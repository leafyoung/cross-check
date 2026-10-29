#!/usr/bin/env python
"""QuantLib golden reference: solves every case in cases/cases.json that has a
'golden' mapping and writes results/quantlib.json. Usage:
    scripts/golden_quantlib.py          (uses cases/cases.json)
"""
import json, math, sys
from pathlib import Path
import QuantLib as ql

ROOT = Path(__file__).resolve().parent.parent
DAY_COUNT = ql.Actual365Fixed()
CAL = ql.NullCalendar()
REF_DATE = ql.Date(26, 10, 2026)
ql.Settings.instance().evaluationDate = REF_DATE


def expiry_date(t):
    return CAL.advance(REF_DATE, int(round(t * 365)), ql.Days)


def flat_curve(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(0, CAL, rate, DAY_COUNT))


def golden_bsm(p):
    spot, strike = p["spot"], p["strike"]
    vol, r, q, t = p["volatility"], p["rate"], p["dividend"], p["expiry"]
    payoff_type = ql.Option.Call if p["option_type"] == "call" else ql.Option.Put
    strike_q = ql.QuoteHandle(ql.SimpleQuote(strike))
    payoff = ql.PlainVanillaPayoff(payoff_type, strike)
    # Black-Scholes-Merton analytic engine on a spot + dividend curve
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)), flat_curve(q), flat_curve(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, CAL, vol, DAY_COUNT)))
    opt = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry_date(t)))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return opt.NPV()


def golden_crr(p):
    spot, strike = p["spot"], p["strike"]
    vol, r, t, steps = p["volatility"], p["rate"], p["expiry"], p["steps"]
    payoff_type = ql.Option.Call if p["option_type"] == "call" else ql.Option.Put
    style_is_american = p.get("option_style", "american") == "american"
    expiry = expiry_date(t)
    style = ql.AmericanExercise(REF_DATE, expiry) if style_is_american else ql.EuropeanExercise(expiry)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)), flat_curve(0.0), flat_curve(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, CAL, vol, DAY_COUNT)))
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(payoff_type, strike), style)
    opt.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
    return opt.NPV()


def golden_heston(p):
    proc = ql.HestonProcess(flat_curve(p["rate"]), flat_curve(p["dividend"]),
                            ql.QuoteHandle(ql.SimpleQuote(p["spot"])),
                            p["v0"], p["kappa"], p["theta"], p["sigma_v"], p["rho"])
    payoff_type = ql.Option.Call if p["option_type"] == "call" else ql.Option.Put
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(payoff_type, p["strike"]),
                           ql.EuropeanExercise(expiry_date(p["expiry"])))
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc)))
    return opt.NPV()


def golden_sabr_iv(p):
    return ql.sabrVolatility(p["strike"], p["f0"], p["expiry"], p["alpha"], p["beta"], p["nu"], p["rho"])


def golden_black76(p):
    payoff_type = ql.Option.Call if p["option_type"] == "call" else ql.Option.Put
    std_dev = p["volatility"] * math.sqrt(p["expiry"])
    discount = math.exp(-p["rate"] * p["expiry"])
    bc = ql.BlackCalculator(ql.PlainVanillaPayoff(payoff_type, p["strike"]),
                            p["forward"], std_dev, discount)
    return bc.value()


def golden_vasicek(p):
    v = ql.Vasicek(p["r0"], p["kappa"], p["theta"], p["sigma"])
    return v.discountBond(0.0, p["maturity"], p["r0"])


def golden_cir(p):
    # NB: QuantLib's CIR ctor order is (r0, theta, kappa, sigma) — unlike
    # Vasicek's (r0, kappa, theta, sigma).
    c = ql.CoxIngersollRoss(p["r0"], p["theta"], p["kappa"], p["sigma"])
    return c.discountBond(0.0, p["maturity"], p["r0"])


def golden_hull_white(p):
    hw = ql.HullWhite(flat_curve(p["r0"]), p["a"], p["sigma"])
    return hw.discountBond(0.0, p["maturity"], p["r0"])


# ---------------------------------------------------------------------------
# Black-Karasinski golden. Two different conventions live under the name
# "BK t=0 ZCB" (README: 'Black-Karasinski golden — two conventions, both
# verified'):
#   - QuantLib: ql.BlackKarasinski recalibrates itself to the initial term
#     structure (the model's theta/x0 is fit to the curve), so its t=0 ZCB is
#     the curve discount factor e^{-r0*T} up to tree discretization error.
#     Route: a CallableZeroCouponBond (zero coupons, empty callability -
#     i.e. a plain ZCB) priced by TreeCallableFixedRateBondEngine off the BK
#     short-rate tree. This is the only model-based bond engine on the pip
#     surface (DiscountingBondEngine is curve-only; BK itself exposes no
#     discountBond - it is not one-factor-affine). Evidence the NPV comes
#     from the tree, not the curve: the finite-step NPV depends on a/sigma
#     (a=0.5, 50 steps: 6.6e-11 below the DF; a=0.1, sigma=0.05: 6e-11 above)
#     and converges to the curve DF as steps grow.
#   - torchquantlib family: numerical trapezoid integral of the expected
#     lognormal short-rate path E[r_t] = exp(ln(r0)e^{-at} +
#     sigma^2(1-e^{-2at})/(2a)) starting from r_t = r0, with NO curve
#     recalibration (_bk_family_zcb below, n=100 points as in the runners).
# The gap (0.8607 curve-DF vs 0.7106 integral, 17% rel) is a convention
# difference, NOT a bug: compare.py renders it CONVENTION via the case's
# "convention" flag in cases.json.
# ---------------------------------------------------------------------------


def _bk_family_zcb(a, sigma, r0, maturity, n):
    """Independent pure-Python port of the torchquantlib family's BK
    zero_coupon_bond_price (quantburn src/black_karasinski.rs):
    exp(-trapezoid of E[r_t] dt), E[r_t] = exp(ln(r0)e^{-at} +
    sigma^2(1-e^{-2at})/(2a)) over n points on [0, maturity]. Pure Python
    floats; reproduces the f64 family runners (quantburn/quantcandle/
    quantjax: 0.7105591541535006 at their n=100) bit-for-bit; quantorch's
    0.7105591893196106 is the same integral in f32 (4.9e-8 rel, f32 tier)."""
    if maturity <= 0.0:
        return 1.0
    a = max(a, 1e-12)
    sigma = max(sigma, 0.0)
    r = max(r0, 1e-8)
    n = max(n, 2)
    dt = maturity / (n - 1)
    integral = 0.0
    for i in range(n):
        t = dt * i
        mean_log_r = math.log(r) * math.exp(-a * t)
        var_log_r = sigma * sigma * (1.0 - math.exp(-2.0 * a * t)) / max(2.0 * a, 1e-12)
        w = 0.5 if (i == 0 or i == n - 1) else 1.0
        integral += w * math.exp(mean_log_r + 0.5 * var_log_r) * dt
    return math.exp(-integral)


def golden_black_karasinski(p):
    """BK t=0 ZCB golden under the QUANTLIB convention: the model is
    recalibrated to the flat r0 curve, so the tree ZCB = curve DF
    e^{-r0*T} up to tree discretization error. Tree-error sweep for the
    case params (a=0.1, sigma=0.01, T=5), rel err vs the DF:
      25 -> 8.5e-13, 50 -> 9.6e-14, 100 -> 1.1e-14, 200 -> 2.6e-15,
      400 -> 5.2e-16, 800 -> 1.3e-15, 1600 -> 1.6e-12 (roundoff starts to
      dominate the discretization), 3200 -> 6.7e-13, 6400 -> 5.8e-13.
    400 steps chosen (measured 5.2e-16); tolerance = measured error x
    ~2000 margin, still far below any real bug scale.
    Cross-asserts (a mismatch raises and surfaces as a golden failure
    rather than being masked):
      - the pure-Python family-semantics reference reproduces the family
        runner value 0.7105591541535006 (n=100) to 1e-12;
      - the tree NPV/face is within 1e-10 of the curve DF."""
    a, sigma, r0, T = p["a"], p["sigma"], p["r0"], p["maturity"]
    fam = _bk_family_zcb(a, sigma, r0, T, 100)
    fam_doc = 0.7105591541535006
    if abs(fam - fam_doc) / fam_doc > 1e-12:
        raise ValueError(f"family BK reference drifted: {fam!r} vs {fam_doc!r}")
    bk = ql.BlackKarasinski(flat_curve(r0), a, sigma)
    mat_date = CAL.advance(REF_DATE, int(round(T * 365)), ql.Days)
    bond = ql.CallableZeroCouponBond(0, 100.0, CAL, mat_date, DAY_COUNT)
    bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(bk, 400))
    zcb = bond.NPV() / 100.0
    df = math.exp(-r0 * (mat_date - REF_DATE) / 365.0)
    if abs(zcb - df) / df > 1e-10:
        raise ValueError(f"BK tree ZCB off the calibrated curve: {zcb!r} vs DF {df!r}")
    return zcb



def _bs_process(spot, r, q, vol):
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)), flat_curve(q), flat_curve(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, CAL, vol, DAY_COUNT)))


def _ql_option(process, payoff_type, strike, expiry, engine):
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if payoff_type == "call" else ql.Option.Put, strike)
    opt = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry_date(expiry)))
    opt.setPricingEngine(engine)
    return opt



def golden_fam_asian_geo(p):
    """Family discrete-geometric Asian (analytical closed form, n monthly
    fixings implied by the model's `steps`) vs QuantLib's exact discrete
    geometric engine. The family's fixings are uniform at j*T/n; QL fixing
    dates round to whole days -> residual date-grid misalignment ~1e-4;
    tolerance 1e-3 documented. NOTE the case omits num_paths so the family
    runners take the analytical path; `steps` in the case params controls n
    (12 monthly fixings). All fam_ asian cases use q=0: the family's
    geometric analytical form has no dividend parameter (upstream API
    limitation — _geometric_asian_price takes no q), so a q>0 case would
    measure that limitation, not the formula; documented in README."""
    process = _bs_process(p["spot"], p["rate"], p["dividend"], p["volatility"])
    n = int(p.get("steps", 12))
    fixing_dates = ql.DateVector(
        [CAL.advance(REF_DATE, int(round(j * p["expiry"] * 365 / n)), ql.Days) for j in range(1, n + 1)])
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if p["option_type"] == "call" else ql.Option.Put, p["strike"])
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Geometric, 0.0, 0, fixing_dates, payoff,
        ql.EuropeanExercise(expiry_date(p["expiry"])))
    option.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))
    return option.NPV()


def golden_tfq_iv(p):
    """Implied-vol roundtrip: golden = the true vol used to generate the
    market price (the price constant in the case was computed with
    BlackCalculator). The runner inverts with tff and must recover it."""
    return p["volatility"]


def golden_tfq_asian_geo(p):
    """Discrete geometric Asian: QuantLib's closed form for the discrete
    geometric average over the same sampling times. tff uses the conditioning
    identity — both exact, tol 1e-6."""
    process = _bs_process(p["spot"], p["rate"], p["dividend"], p["volatility"])
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if p["option_type"] == "call" else ql.Option.Put, p["strike"])
    fixing_dates = ql.DateVector(
        [CAL.advance(REF_DATE, int(round(t * 365)), ql.Days) for t in p["sampling_times"]])
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Geometric, 0.0, 0, fixing_dates, payoff,
        ql.EuropeanExercise(expiry_date(p["expiry"])))
    option.setPricingEngine(
        ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))
    return option.NPV()


def golden_tfq_asian_arith_lb(p):
    """Arithmetic discrete Asian: tff returns a LOWER BOUND (Curran-style
    conditioning); golden is QuantLib MC on the same sampling grid. The bound
    sits below the true price and the MC estimate carries sampling error, so
    the tolerance is loose (3e-2) with one-sidedness documented, not masked."""
    process = _bs_process(p["spot"], p["rate"], p["dividend"], p["volatility"])
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if p["option_type"] == "call" else ql.Option.Put, p["strike"])
    fixing_dates = ql.DateVector(
        [CAL.advance(REF_DATE, int(round(t * 365)), ql.Days) for t in p["sampling_times"]])
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic, 0.0, 0, fixing_dates, payoff,
        ql.EuropeanExercise(expiry_date(p["expiry"])))
    option.setPricingEngine(ql.MCDiscreteArithmeticAPEngine(
        process, "pseudorandom", brownianBridge=False, antitheticVariate=False,
        controlVariate=False, requiredSamples=200000, seed=42))
    return option.NPV()


def golden_tfq_barrier(p):
    """Continuous-monitoring single barrier, rebate=0: tff's analytic
    formulas vs QuantLib AnalyticBarrierEngine — both exact, tol 1e-6."""
    process = _bs_process(p["spot"], p["rate"], p["dividend"], p["volatility"])
    barrier_type = {"down-and-out": ql.Barrier.DownOut,
                    "up-and-out": ql.Barrier.UpOut,
                    "down-and-in": ql.Barrier.DownIn,
                    "up-and-in": ql.Barrier.UpIn}[p["barrier_type"]]
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if p["option_type"] == "call" else ql.Option.Put, p["strike"])
    option = ql.BarrierOption(barrier_type, p["barrier"], 0.0, payoff,
                              ql.EuropeanExercise(expiry_date(p["expiry"])))
    option.setPricingEngine(ql.AnalyticBarrierEngine(process))
    return option.NPV()


def golden_tfq_binary(p):
    """Cash-or-nothing binary (pays 1 unit of cash): tff binary_price vs
    QuantLib AnalyticEuropeanEngine on CashOrNothingPayoff — both exact."""
    process = _bs_process(p["spot"], p["rate"], p["dividend"], p["volatility"])
    payoff = ql.CashOrNothingPayoff(
        ql.Option.Call if p["option_type"] == "call" else ql.Option.Put,
        p["strike"], 1.0)
    opt = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry_date(p["expiry"])))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return opt.NPV()


def golden_tfq_pde(p):
    """Vanilla European via finite differences: golden from QuantLib's
    FD engine (same class of discretization error; tol 5e-3 documented)."""
    process = _bs_process(p["spot"], p["rate"], p["dividend"], p["volatility"])
    return _ql_option(process, p["option_type"], p["strike"], p["expiry"],
                      ql.FdBlackScholesVanillaEngine(process, 100, 400)).NPV()


def golden_tfq_swapcurve(p):
    """Zero-curve bootstrap on a shared basis: par swaps generated from a
    flat continuously-compounded curve; tff's default interpolator is linear
    in zero rates, so golden uses PiecewiseLinearZero. Returns the dict of
    discount factors at the case's test maturities."""
    r_flat = p["flat_rate"]
    ref = REF_DATE
    index = ql.IborIndex("TfqIdx", ql.Period(6, ql.Months), 0, ql.USDCurrency(),
                         ql.NullCalendar(), ql.Following, False, DAY_COUNT)

    def par_rate(T):
        float_leg = 0.0
        for k in range(1, int(round(T / 0.5)) + 1):
            t1, t2 = 0.5 * (k - 1), 0.5 * k
            float_leg += 0.5 * (math.exp(r_flat * (t2 - t1)) - 1.0) * math.exp(-r_flat * t2)
        annuity = sum(math.exp(-r_flat * k) for k in range(1, int(round(T)) + 1))
        return float_leg / annuity

    helpers = []
    for T in p["swap_maturities"]:
        helpers.append(ql.SwapRateHelper(
            par_rate(T), ql.Period(int(round(T)), ql.Years),
            CAL, ql.Annual, ql.Unadjusted, DAY_COUNT, index))
    curve = ql.PiecewiseLinearZero(ref, helpers, DAY_COUNT)
    curve.enableExtrapolation()
    return {"dfs": [curve.discount(t) for t in p["test_maturities"]]}


# ---------------------------------------------------------------------------
# Independent pure-Python Hagan-West reference (monotone convex interpolation
# + bond-curve bootstrap), written from Hagan & West 2006/2008. Used as the
# golden for tfqr_hw_* because QuantLib has no monotone-convex (Hagan-West)
# curve builder on its pip surface, and its log-linear-discount bootstrap
# differs in basis for coupon instruments (see README).
#
# Verification of this reference (all performed at development time):
#   - interval-integral exactness Integral[f] = fd*Dt holds to 1e-17 over
#     random sweeps of all scheme regions (incl. negative forwards);
#   - endpoint continuity to 5e-17;
#   - reproduces the HW 2008 paper bootstrap table (the tfqr_hw_bondcurve
#     case: [5.00, 4.75, 4.53333333, 4.775]%) to 9e-14 and re-prices the
#     input bonds to 3e-12;
#   - a single bootstrap step matches tff's bond_curve one_step digit for
#     digit on the paper example.
# ---------------------------------------------------------------------------


def _hw_endpoint_forwards(ts, fd):
    """Node instantaneous forwards from interval forwards fd_i on
    [t_{i-1}, t_i] (t_{-1}=0): straight line through adjacent interval
    midpoints at interior nodes; half-reflection end rule
    f_0 = fd_1 - (f_1 - fd_1)/2 (mirrored at the far end), the end treatment
    documented in tff's implementation."""
    n = len(fd)
    f = [0.0] * (n + 1)
    for i in range(n - 1):
        left = 0.0 if i == 0 else ts[i - 1]
        m_i = (left + ts[i]) / 2.0
        m_ip1 = (ts[i] + ts[i + 1]) / 2.0
        slope = (fd[i + 1] - fd[i]) / (m_ip1 - m_i)
        f[i + 1] = fd[i] + slope * (ts[i] - m_i)
    f[0] = fd[0] - 0.5 * (f[1] - fd[0])
    f[n] = fd[n - 1] - 0.5 * (f[n - 1] - fd[n - 1])
    return f


def _hw_interval(g0, g1, x):
    """Monotone-convex deviation curve on one interval: returns (dev(x),
    integral of dev over [0, x]); the integral over [0,1] vanishes by
    construction. Region map per Hagan-West (Fig. 4 of the 2008 paper)."""
    s = g1 + 2.0 * g0
    r = g0 + 2.0 * g1
    if (s < 0.0 <= r) or (s > 0.0 >= r):
        dev = (3.0 * (g0 + g1) * x - (2.0 * g1 + 4.0 * g0)) * x + g0
        integ = x * (1.0 - x) * (g0 * (1.0 - x) - g1 * x)
        return dev, integ
    if (g0 < 0.0 and s >= 0.0) or (g0 > 0.0 and s <= 0.0):
        eta = s / (g1 - g0)
        xf = x if x > eta else eta
        ratio = (xf - eta) / (1.0 - eta)
        return (g0 + (g1 - g0) * ratio ** 2,
                g0 * x + (g1 - g0) * (1.0 - eta) / 3.0 * ratio ** 3)
    if (g1 <= 0.0 < r) or (g1 >= 0.0 > r):
        eta = 3.0 * g1 / (g1 - g0)
        xc = x if x < eta else eta
        ratio = (eta - xc) / eta
        return (g1 + (g0 - g1) * ratio ** 2,
                g1 * x + eta * (g0 - g1) / 3.0 * (1.0 - ratio ** 3))
    eta = g1 / (g0 + g1)
    shift = -0.5 * (eta * g0 + (1.0 - eta) * g1)
    xc = x if x < eta else eta
    xf = x if x > eta else eta
    ratio_c = (eta - xc) / eta
    ratio_f = (xf - eta) / (1.0 - eta)
    return (shift + (g0 - shift) * ratio_c ** 2 + (g1 - shift) * ratio_f ** 2,
            shift * x + (g0 - shift) * eta * (1.0 - ratio_c ** 3) / 3.0 +
            (g1 - shift) * (1.0 - eta) * ratio_f ** 3 / 3.0)


def _hw_locate(t, ts):
    n = len(ts)
    if t <= ts[0]:
        return 0
    if t >= ts[n - 1]:
        return n - 1
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t <= ts[mid]:
            hi = mid
        else:
            lo = mid
    return hi


def _hw_interpolate_forward(ts, fd, times):
    """Instantaneous forward and its running integral (from the containing
    interval's left end) at the query times."""
    f = _hw_endpoint_forwards(ts, fd)
    out = []
    for t in times:
        k = _hw_locate(t, ts)
        tl = 0.0 if k == 0 else ts[k - 1]
        dt = ts[k] - tl
        x = (t - tl) / dt
        dev, integ = _hw_interval(f[k] - fd[k], f[k + 1] - fd[k], x)
        out.append((fd[k] + dev, (integ + fd[k] * x) * dt))
    return out


def _hw_forwards_from_yields(ts, ys):
    fd, prev_acc = [], 0.0
    for i in range(len(ys)):
        acc = ys[i] * ts[i]
        tl = 0.0 if i == 0 else ts[i - 1]
        fd.append((acc - prev_acc) / (ts[i] - tl))
        prev_acc = acc
    return fd


def _hw_interpolate_yields(ts, ys, times):
    fd = _hw_forwards_from_yields(ts, ys)
    res = _hw_interpolate_forward(ts, fd, times)
    out = []
    for t, (_, integ) in zip(times, res):
        k = _hw_locate(t, ts)
        base = 0.0 if k == 0 else ys[k - 1] * ts[k - 1]
        out.append((base + integ) / t if t > 0 else 0.0)
    return out


def _hw_yields_from_forwards(ts, fd):
    y, acc = [], 0.0
    for i in range(len(fd)):
        tl = 0.0 if i == 0 else ts[i - 1]
        acc += fd[i] * (ts[i] - tl)
        y.append(acc / ts[i])
    return y


def _hw_yield_from_pv(cfs, ts, pv, tol=1e-12, maxit=100):
    y = 0.0
    for _ in range(maxit):
        p = sum(c * math.exp(-y * t) for c, t in zip(cfs, ts))
        dur = sum(t * c * math.exp(-y * t) for c, t in zip(cfs, ts))
        dy = (p - pv) / dur
        y += dy
        if abs(dy) < tol:
            break
    return y


def _hw_bootstrap(cashflows, cftimes, pvs, tol=1e-12, maxit=100):
    """Hagan-West 2008 iterative bootstrap: interpolate the current yield
    curve to all cashflow times with monotone convex, invert the bond pricing
    equation for the expiry discount factors, iterate."""
    expiry = [t[-1] for t in cftimes]
    n = len(cashflows)
    rates = [_hw_yield_from_pv(cashflows[i], cftimes[i], pvs[i]) for i in range(n)]
    calc_ts, calc_cs, groups = [], [], []
    for i in range(n):
        calc_ts.append(0.0)
        calc_cs.append(-pvs[i])
        calc_ts.extend(cftimes[i][:-1])
        calc_cs.extend(cashflows[i][:-1])
        groups.extend([i] * len(cftimes[i]))
    for _ in range(maxit):
        dfs = [math.exp(-rates[i] * expiry[i]) for i in range(n)]
        rates_at = _hw_interpolate_yields(expiry, rates, calc_ts)
        discounts = [math.exp(-r * t) for r, t in zip(rates_at, calc_ts)]
        new_dfs = []
        for i in range(n):
            s = sum(calc_cs[j] * discounts[j]
                    for j, g in enumerate(groups) if g == i)
            new_dfs.append(-s / cashflows[i][-1])
        new_rates = [-math.log(z) / t for z, t in zip(new_dfs, expiry)]
        if max(abs(a - b) for a, b in zip(dfs, new_dfs)) < tol:
            rates = new_rates
            break
        rates = new_rates
    return expiry, rates


def _nss_yield(t, prm):
    """Nelson-Siegel-Svensson zero yield (Muller, BIS Papers No 25)."""
    h1 = (1.0 - math.exp(-t / prm["tau1"])) / (t / prm["tau1"])
    h2 = (1.0 - math.exp(-t / prm["tau2"])) / (t / prm["tau2"])
    return (prm["beta0"] + prm["beta1"] * h1 +
            prm["beta2"] * (h1 - math.exp(-t / prm["tau1"])) +
            prm["beta3"] * (h2 - math.exp(-t / prm["tau2"])))


def golden_tfqr_hw_bondcurve(p):
    """Hagan-West bond-curve bootstrap vs an INDEPENDENT pure-Python
    Hagan-West reference (this file's _hw_* code). QuantLib has no
    monotone-convex builder (README: QL-vs-HW scheme difference), and for
    coupon instruments the two bootstraps interpolate on different bases
    (forwards vs log-discounts), so golden-vs-QL is not meaningful here; the
    reference was verified against the published HW 2008 example table
    (which this case's bonds reproduce: [5.00, 4.75, 4.5333, 4.775]%)."""
    expiry, rates = _hw_bootstrap(p["bond_cashflows"], p["bond_cashflow_times"],
                                  p["present_values"])
    return {"rates": rates, "dfs": [math.exp(-r * t) for r, t in zip(rates, expiry)]}


def golden_tfqr_hw_monotone(p):
    """Monotone-convex interpolation (forwards and yields) vs the independent
    pure-Python reference. Includes the negative-forward curve that drives
    the scheme into its region-2/3 branches."""
    yc = p["yields_curve"]
    y_vals = _hw_interpolate_yields(yc["reference_times"], yc["yields"],
                                    yc["query_times"])
    fc = p["fwd_curve"]
    f_vals = [v for v, _ in _hw_interpolate_forward(
        fc["reference_times"], fc["forwards"], fc["query_times"])]
    fn = p["fwd_curve_neg"]
    fn_vals = [v for v, _ in _hw_interpolate_forward(
        fn["reference_times"], fn["forwards"], fn["query_times"])]
    yn_vals = _hw_interpolate_yields(
        fn["reference_times"],
        _hw_yields_from_forwards(fn["reference_times"], fn["forwards"]),
        fn["query_times"])
    return {"forwards": f_vals, "yields": y_vals,
            "fwd_neg": fn_vals, "yld_neg": yn_vals}


def golden_tfqr_hw_zcb(p):
    """Hagan-West vs QuantLib PiecewiseLogLinearDiscount on the shared basis
    where both schemes are pinned: a zero-coupon strip (node DF = price/face
    exactly, independent of the interpolation scheme). Cross-verified in
    here: the QL deposit-bootstrapped curve reproduces the true DFs at the
    node maturities to <=1e-10 (measured 1.2e-13); a mismatch raises and
    surfaces as a golden failure rather than being masked."""
    prm = p["nss_params"]
    face = p["face"]
    days = p["zcb_days"]
    dfs = [math.exp(-_nss_yield(d / 365.0, prm) * d / 365.0) for d in days]
    helpers = []
    for d, df in zip(days, dfs):
        t = d / 365.0
        rdep = (1.0 / df - 1.0) / t  # simple-compounded deposit rate
        helpers.append(ql.DepositRateHelper(
            rdep, ql.Period(d, ql.Days), 0, CAL, ql.Unadjusted, False,
            DAY_COUNT))
    curve = ql.PiecewiseLogLinearDiscount(REF_DATE, helpers, DAY_COUNT)
    curve.enableExtrapolation()
    worst = max(abs(curve.discount(d / 365.0) - df) / df
                for d, df in zip(days, dfs))
    if worst > 1e-10:
        raise ValueError(f"QL ZCB bootstrap mismatch: {worst:.3e}")
    return {"dfs": dfs}


def golden_tfqr_nss_fit(p):
    """Nelson-Siegel-Svensson: golden = the true generating curve at the test
    maturities (synthetic data lies exactly on the model manifold, and the
    test points are fit tenors, so the target is independent of the
    non-unique fitted parameters). Cross-verified in here: QuantLib's
    SvenssonFitting on zero-coupon bonds priced off the true curve recovers
    the curve at the tenors to <=1e-9 (measured 6e-13); a mismatch raises."""
    prm = p["true_params"]
    tenors = p["fit_tenors"]
    helpers = []
    for t in tenors:
        y = _nss_yield(t, prm)
        mat = CAL.advance(REF_DATE, int(round(365 * t)), ql.Days)
        helpers.append(ql.BondHelper(
            ql.QuoteHandle(ql.SimpleQuote(100.0 * math.exp(-y * t))),
            ql.ZeroCouponBond(0, CAL, 100.0, mat)))
    curve = ql.FittedBondDiscountCurve(REF_DATE, helpers, DAY_COUNT,
                                       ql.SvenssonFitting())
    curve.enableExtrapolation()
    _ = curve.discount(1.0)  # trigger the fit
    worst = 0.0
    for t in tenors:
        z = -math.log(curve.discount(t)) / t
        worst = max(worst, abs(z - _nss_yield(t, prm)))
    if worst > 1e-9:
        raise ValueError(f"QL Svensson fit did not recover the curve: {worst:.3e}")
    return {"yields": [_nss_yield(tenors[i], prm) for i in p["test_indices"]]}


def golden_tfqr_forwards(p):
    """Forward/yield identities vs closed-form pure Python: consecutive
    forwards from yields f_i = (y_i t_i - y_{i-1} t_{i-1})/(t_i - t_{i-1}),
    the inverse mapping, and forwards from discount factors
    (df_start/df_end - 1)/dcf."""
    prm = p["nss_params"]
    ts = p["times"]
    ys = [_nss_yield(t, prm) for t in ts]
    fwd_ref = [ys[0]]
    for i in range(1, len(ts)):
        fwd_ref.append((ys[i] * ts[i] - ys[i - 1] * ts[i - 1]) / (ts[i] - ts[i - 1]))
    dfs = [math.exp(-y * t) for y, t in zip(ys, ts)]
    df_fwd = [(dfs[i] / dfs[i + 1] - 1.0) / (ts[i + 1] - ts[i])
              for i in range(len(ts) - 1)]
    return {"fwd": fwd_ref, "roundtrip": ys, "df_fwd": df_fwd}


def golden_tfqr_cashflows(p):
    """Cashflow analytics vs closed form: PV of a coupon stream at a flat
    continuously-compounded yield (direct sum), Newton yield recovery
    (target = the input yield), and PV from explicit discount factors."""
    cfs = [p["coupon"]] * (len(p["cashflow_times"]) - 1) + [p["final_cashflow"]]
    ts, y = p["cashflow_times"], p["flat_yield"]
    pv_ref = sum(c * math.exp(-y * t) for c, t in zip(cfs, ts))
    dfs = [math.exp(-y * t) for t in ts]
    pv_dfs = sum(c * d for c, d in zip(cfs, dfs))
    return {"pv": [pv_ref], "yield": [y], "pv_dfs": [pv_dfs]}


def golden_tfqm_digital(p):
    """Digital (cash-or-nothing) golden: exact QuantLib analytic on
    CashOrNothingPayoff - preferred over an MC golden because the closed
    form exists. Verified against the pure-Python closed form
    payout*e^{-rT}*N(d2) (agrees to 2e-15) and a 500k-sample terminal-GBM
    Monte Carlo (5.3258 +- 0.0067). The family runners price this by MC with
    256 paths (rel SE ~5.6%), so their rows are compared at the rng
    tolerance (PASS(rng)/INFO(rng)) in compare.py, never silently."""
    process = _bs_process(p["spot"], p["rate"], p.get("dividend", 0.0),
                          p["volatility"])
    payoff = ql.CashOrNothingPayoff(
        ql.Option.Call if p["option_type"] == "call" else ql.Option.Put,
        p["strike"], p["payout"])
    opt = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry_date(p["expiry"])))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return opt.NPV()


def golden_tfqm_lookback(p):
    """Fixed-strike lookback call golden: exact QuantLib analytic
    (Conze-Viswanathan) with CONTINUOUS monitoring - pip QuantLib 1.44 does
    expose AnalyticContinuousFixedLookbackEngine, so no SKIP is needed.
    Verified against the pure-Python Conze-Viswanathan formula (agrees to
    2e-14) and a 500k-sample, 252-step MC (18.375 +- 0.022 vs 19.1676 here:
    the family's DISCRETE 252-step monitoring sits ~4.1% below the
    continuous-monitoring value, matching the Broadie-Glasserman-Kou
    adjustment e^{-0.5826*sigma*sqrt(dt)} ~ -0.73% on the running max; see
    README). MC rows therefore compare at the rng tolerance only."""
    S, K = p["spot"], p["strike"]
    vol, r, t = p["volatility"], p["rate"], p["expiry"]
    process = _bs_process(S, r, p.get("dividend", 0.0), vol)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if p["option_type"] == "call" else ql.Option.Put, K)
    opt = ql.ContinuousFixedLookbackOption(
        S, payoff, ql.EuropeanExercise(expiry_date(t)))
    opt.setPricingEngine(ql.AnalyticContinuousFixedLookbackEngine(process))
    return opt.NPV()

GOLDEN = {
    "bsm": golden_bsm,
    "crr": golden_crr,
    "heston": golden_heston,
    "sabr_iv": golden_sabr_iv,
    "black76": golden_black76,
    "vasicek": golden_vasicek,
    "cir": golden_cir,
    "hull_white": golden_hull_white,
    "black_karasinski": golden_black_karasinski,
    "tfq_iv": golden_tfq_iv,
    "tfq_iv_approx": golden_tfq_iv,
    "fam_asian_geo": golden_fam_asian_geo,
    "tfq_asian_geo": golden_tfq_asian_geo,
    "tfq_asian_arith_lb": golden_tfq_asian_arith_lb,
    "tfq_barrier": golden_tfq_barrier,
    "tfq_binary": golden_tfq_binary,
    "tfq_pde": golden_tfq_pde,
    "tfq_swapcurve": golden_tfq_swapcurve,
    "tfqr_hw_bondcurve": golden_tfqr_hw_bondcurve,
    "tfqr_hw_monotone": golden_tfqr_hw_monotone,
    "tfqr_hw_zcb": golden_tfqr_hw_zcb,
    "tfqr_nss_fit": golden_tfqr_nss_fit,
    "tfqr_forwards": golden_tfqr_forwards,
    "tfqr_cashflows": golden_tfqr_cashflows,
    "tfqm_digital": golden_tfqm_digital,
    "tfqm_lookback": golden_tfqm_lookback,
}

# Golden entries returning a dict of lists (vectors) are compared per-key by
# compare.py (max rel err over all keys/values).

# crr: torchquantlib family uses classic multiplicative CRR; QuantLib's
# CoxRossRubinstein lattice is the equal-jumps drift-centered variant. Same
# sigma*sqrt(dt) jump, different node placement -> O(1/steps) discretization
# bias between them (~6e-5 at 100 steps). 1e-4 still catches real bugs
# (e.g. the tf-quant discount-order bug was 5e-4).
# tfq_iv_approx: Stefanica-Radiocic closed-form approximation (~6e-4 near ATM).
# tfq_pde: FD discretization on both sides (tff uniform grid vs QL FD engine).
# tfqr_hw_bondcurve: both sides stop on the same 1e-8-discount-factor
#   iteration criterion from slightly different iterates; the residual
#   between two stopping states is O(1e-7) on DFs / O(1e-6) on rates
#   (measured ~1e-9; headroom x1000). Both re-price the inputs to <1e-9.
# tfqr_hw_monotone: identical closed-form scheme, independent code paths;
#   measured agreement ~1e-15, 1e-9 leaves f32-vs-f64 headroom.
# tfqr_hw_zcb / tfqr_nss_fit / tfqr_forwards / tfqr_cashflows: exact
#   identities or machine-precision recovery on pinned quantities
#   (measured 1e-16 .. 1e-12); 1e-9 is pure headroom.
# tfqm_digital / tfqm_lookback: analytic goldens; the compared rows are MC
#   estimates from different RNG streams (256/128 paths), so compare.py
#   applies the 1e-2 rng tolerance with PASS(rng)/INFO(rng) verdicts.
# black_karasinski: BK-tree discretization vs the calibrated curve DF;
#   measured 5.2e-16 at the chosen 400 steps (sweep in golden function doc);
#   1e-12 = measured x ~2000 margin. The family rows exceed ANY tolerance
#   here by construction (different convention) and render CONVENTION.
GOLDEN_TOL = {"heston": 1e-5, "crr": 1e-4,
              "tfq_iv_approx": 1e-3, "tfq_pde": 5e-3,
              "tfqr_hw_bondcurve": 1e-6, "tfqr_hw_monotone": 1e-9,
              "tfqr_hw_zcb": 1e-9, "tfqr_nss_fit": 1e-9,
              "fam_asian_geo": 1e-3,
              "tfqr_forwards": 1e-9, "tfqr_cashflows": 1e-8,
              "tfqm_digital": 1e-2, "tfqm_lookback": 1e-2,
              "black_karasinski": 1e-12}  # relative, else 1e-6


def main():
    cases = json.loads((ROOT / "cases" / "cases.json").read_text())["cases"]
    results = []
    for c in cases:
        entry = {"id": c["id"], "ok": False}
        try:
            g = c.get("golden")
            if g is None:
                entry.update(ok=True, price=None, skipped=True, note="no QuantLib equivalent")
            else:
                entry.update(ok=True, price=GOLDEN[g](c["params"]), golden=g,
                             tol=GOLDEN_TOL.get(g, 1e-6))
        except Exception as e:
            entry.update(ok=False, error=str(e))
        results.append(entry)
    out = {"backend": "quantlib", "results": results}
    (ROOT / "results" / "quantlib.json").write_text(json.dumps(out, indent=2))
    solved = sum(1 for r in results if r["ok"] and r.get("price") is not None)
    print(f"golden: {solved} priced, {sum(1 for r in results if r.get('skipped'))} skipped, "
          f"{sum(1 for r in results if not r['ok'])} failed -> results/quantlib.json")


if __name__ == "__main__":
    sys.exit(main())
