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
    discount factors at the case's test maturities (GOLDEN_SHAPE 'dfs')."""
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

GOLDEN = {
    "bsm": golden_bsm,
    "crr": golden_crr,
    "heston": golden_heston,
    "sabr_iv": golden_sabr_iv,
    "black76": golden_black76,
    "vasicek": golden_vasicek,
    "cir": golden_cir,
    "hull_white": golden_hull_white,
    "tfq_iv": golden_tfq_iv,
    "tfq_iv_approx": golden_tfq_iv,
    "tfq_asian_geo": golden_tfq_asian_geo,
    "tfq_asian_arith_lb": golden_tfq_asian_arith_lb,
    "tfq_barrier": golden_tfq_barrier,
    "tfq_binary": golden_tfq_binary,
    "tfq_pde": golden_tfq_pde,
    "tfq_swapcurve": golden_tfq_swapcurve,
}

# Golden entries that return non-scalar payloads (e.g. dict of DFs) are
# compared specially in compare.py via GOLDEN_SHAPE below.
GOLDEN_SHAPE = {"tfq_swapcurve": "dfs"}

# crr: torchquantlib family uses classic multiplicative CRR; QuantLib's
# CoxRossRubinstein lattice is the equal-jumps drift-centered variant. Same
# sigma*sqrt(dt) jump, different node placement -> O(1/steps) discretization
# bias between them (~6e-5 at 100 steps). 1e-4 still catches real bugs
# (e.g. the tf-quant discount-order bug was 5e-4).
# tfq_iv_approx: Stefanica-Radiocic closed-form approximation (~6e-4 near ATM).
# tfq_pde: FD discretization on both sides (tff uniform grid vs QL FD engine).
GOLDEN_TOL = {"heston": 1e-5, "crr": 1e-4,
              "tfq_iv_approx": 1e-3, "tfq_pde": 5e-3}  # relative, else 1e-6


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
