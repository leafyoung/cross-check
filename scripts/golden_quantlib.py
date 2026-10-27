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


GOLDEN = {
    "bsm": golden_bsm,
    "crr": golden_crr,
    "heston": golden_heston,
    "sabr_iv": golden_sabr_iv,
    "black76": golden_black76,
    "vasicek": golden_vasicek,
    "cir": golden_cir,
    "hull_white": golden_hull_white,
}

# crr: torchquantlib family uses classic multiplicative CRR; QuantLib's
# CoxRossRubinstein lattice is the equal-jumps drift-centered variant. Same
# sigma*sqrt(dt) jump, different node placement -> O(1/steps) discretization
# bias between them (~6e-5 at 100 steps). 1e-4 still catches real bugs
# (e.g. the tf-quant discount-order bug was 5e-4).
GOLDEN_TOL = {"heston": 1e-5, "crr": 1e-4}  # relative, else default 1e-6


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
