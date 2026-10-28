#!/usr/bin/env python
"""Calibration cross-check: quantorch vs QuantLib on synthetic data with known ground truth.

Generates a synthetic vanilla-option surface from KNOWN Heston parameters using
QuantLib's AnalyticHestonEngine (independent generator; avoids quantorch's own
pricer biasing the fit), then calibrates:

  1. quantorch's Heston calibrator (torchquantlib.calibration.option_surface_calibration,
     Adam on price-space MSE, constrained-param clamps), started away from truth;
  2. QuantLib's HestonModel + HestonModelHelper + LevenbergMarquardt, same start;
  3. (SABR) quantorch SABR calibration vs ql.sabrVolatility Hagan surfaces; QL side
     fit with a pure-Python Nelder-Mead on ql.sabrVolatility residuals (QuantLib-Python
     exposes no direct SABR calibrator; scipy/numpy are unavailable in the QL venv).

Usage (two venvs, phases orchestrated externally or via `all`):
  cross-check/.venv/bin/python calibration_check.py generate   # QuantLib venv
  quantorch/.venv/bin/python  calibration_check.py quantorch    # torch venv
  cross-check/.venv/bin/python calibration_check.py qlfit       # QuantLib venv
  either                        calibration_check.py report

Intermediates live in /tmp; final artifact: results/calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
QUANTORCH_ROOT = "/var/home/yangye/devv/qfin/pricing/quantorch"

TMP_SURFACE = "/tmp/calib_surface.json"
TMP_TORCH = "/tmp/calib_quantorch.json"
TMP_QL = "/tmp/calib_ql.json"
RESULTS_PATH = os.path.join(REPO, "results", "calibration.json")

# ----------------------------------------------------------------------------
# Ground truth / configuration
# ----------------------------------------------------------------------------

HESTON_TRUTH = {"kappa": 1.0, "theta": 0.04, "sigma_v": 0.5, "rho": -0.5, "v0": 0.04}
HESTON_START = {"kappa": 0.5, "theta": 0.06, "sigma_v": 0.8, "rho": -0.3, "v0": 0.06}
S0, RATE, DIV = 100.0, 0.05, 0.0
STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]
MAT_DAYS = [91, 183, 365, 730]  # ACT/365F -> T ~ 0.249, 0.501, 1.0, 2.0

# beta = 0.5 with F = 100 => alpha is on the "absolute" scale; alpha = 2.0 gives
# ATM lognormal vol ~ 20%. beta is held fixed at truth in the SABR fits (standard
# practice: beta is weakly identified from a single-smile family).
SABR_TRUTH = {"alpha": 2.0, "beta": 0.5, "rho": -0.25, "nu": 0.6}
SABR_START = {"alpha": 1.5, "beta": 0.5, "rho": -0.10, "nu": 0.90}
FWD = 100.0
SABR_DAYS = [183, 365, 730]

TORCH_HESTON_SCHEDULE = [(0.05, 2000), (0.01, 2000)]  # lr annealing via repeated API calls
TORCH_SABR_EPOCHS = 1500
TORCH_LR = 0.05


def rel_err(recovered: float, truth: float) -> float:
    if truth == 0.0:
        return abs(recovered - truth)
    return abs(recovered - truth) / abs(truth)


# ----------------------------------------------------------------------------
# Phase: generate synthetic surface with QuantLib (cross-check venv)
# ----------------------------------------------------------------------------


def phase_generate() -> None:
    import QuantLib as ql

    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    cal = ql.NullCalendar()
    dc = ql.Actual365Fixed()

    spot_q = ql.SimpleQuote(S0)
    spot_h = ql.QuoteHandle(spot_q)
    rts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(RATE)), dc, ql.Continuous, ql.Annual)
    )
    qts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(DIV)), dc, ql.Continuous, ql.Annual)
    )

    t = HESTON_TRUTH
    proc = ql.HestonProcess(rts, qts, spot_h, t["v0"], t["kappa"], t["theta"], t["sigma_v"], t["rho"])
    model = ql.HestonModel(proc)
    engine = ql.AnalyticHestonEngine(model, 144)

    heston_quotes = []
    for nd in MAT_DAYS:
        mat = cal.advance(today, ql.Period(nd, ql.Days))
        T = dc.yearFraction(today, mat)
        disc = rts.discount(mat)
        divd = qts.discount(mat)
        fwd = S0 * divd / disc
        ex = ql.EuropeanExercise(mat)
        for K in STRIKES:
            opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ex)
            opt.setPricingEngine(engine)
            px = opt.NPV()
            sd = ql.blackFormulaImpliedStdDev(ql.Option.Call, K, fwd, px, disc, 0.0, 0.2, 1e-10, 200)
            heston_quotes.append(
                {"K": K, "T": T, "days": nd, "price": px, "iv": sd / math.sqrt(T), "option_type": "call"}
            )

    s = SABR_TRUTH
    sabr_quotes = []
    for nd in SABR_DAYS:
        T = nd / 365.0
        for K in STRIKES:
            vol = ql.sabrVolatility(K, FWD, T, s["alpha"], s["beta"], s["nu"], s["rho"])
            sabr_quotes.append({"K": K, "T": T, "vol": vol})

    payload = {"heston": heston_quotes, "sabr": sabr_quotes}
    with open(TMP_SURFACE, "w") as fh:
        json.dump(payload, fh)
    print(f"[generate] wrote {TMP_SURFACE}: {len(heston_quotes)} Heston quotes, {len(sabr_quotes)} SABR quotes")
    print(f"[generate] sample Heston: {heston_quotes[0]}")


# ----------------------------------------------------------------------------
# Phase: calibrate with quantorch (quantorch venv)
# ----------------------------------------------------------------------------


def phase_quantorch() -> None:
    if QUANTORCH_ROOT not in sys.path:
        sys.path.insert(0, QUANTORCH_ROOT)

    import torch

    torch.manual_seed(1234)

    from torchquantlib.calibration import OptionSurfaceQuote, calibrate_option_surface
    from torchquantlib.models.stochastic_volatility.heston import Heston
    from torchquantlib.models.stochastic_volatility.sabr import SABR

    surface = json.load(open(TMP_SURFACE))

    # ---- Heston (price-space targets) ----
    heston_quotes = [
        OptionSurfaceQuote(
            strike=row["K"],
            expiry=row["T"],
            target=row["price"],
            option_type="call",
            rate=RATE,
            dividend=DIV,
            spot=S0,
            target_type="price",
            name=f"call_K{row['K']:g}_T{row['T']:.4f}",
        )
        for row in surface["heston"]
    ]
    start = HESTON_START
    heston_model = Heston(
        kappa_init=start["kappa"],
        theta_init=start["theta"],
        sigma_v_init=start["sigma_v"],
        rho_init=start["rho"],
        v0_init=start["v0"],
    )
    t0 = time.perf_counter()
    heston_cal = None
    stage_losses = []
    for stage_lr, stage_epochs in TORCH_HESTON_SCHEDULE:
        heston_cal = calibrate_option_surface(
            heston_model, heston_quotes, num_epochs=stage_epochs, lr=stage_lr, verbose=False
        )
        stage_losses.append({"lr": stage_lr, "epochs": stage_epochs, "final_loss": float(heston_cal.result.final_loss)})
    heston_seconds = time.perf_counter() - t0
    params = heston_cal.result.params
    heston_out = {
        "engine": "quantorch/MarketCalibrator (Adam, price-space MSE, lr-annealed)",
        "start": {k: float(v) for k, v in start.items()},
        "recovered": {name: float(params[name]) for name in ("kappa", "theta", "sigma_v", "rho", "v0")},
        "final_loss": float(heston_cal.result.final_loss),
        "price_rmse": float(heston_cal.result.residual_summary["rmse"]),
        "price_max_abs_error": float(heston_cal.result.residual_summary["max_abs"]),
        "stages": stage_losses,
        "duration_s": heston_seconds,
    }
    print(f"[quantorch] Heston fit done in {heston_seconds:.1f}s: {heston_out['recovered']}")
    print(f"[quantorch] Heston final_loss={heston_out['final_loss']:.3e} price_rmse={heston_out['price_rmse']:.4e}")

    # ---- SABR (implied-vol-space targets, beta frozen at truth) ----
    sabr_quotes = [
        OptionSurfaceQuote(
            strike=row["K"],
            expiry=row["T"],
            target=row["vol"],
            rate=0.0,
            target_type="implied_vol",
            model="lognormal",
            name=f"vol_K{row['K']:g}_T{row['T']:.3f}",
        )
        for row in surface["sabr"]
    ]
    sstart = SABR_START
    sabr_model = SABR(
        alpha_init=sstart["alpha"], beta_init=sstart["beta"], rho_init=sstart["rho"], nu_init=sstart["nu"], F0=FWD
    )
    sabr_model.params["beta"].requires_grad_(False)
    t0 = time.perf_counter()
    sabr_cal = calibrate_option_surface(
        sabr_model, sabr_quotes, num_epochs=TORCH_SABR_EPOCHS, lr=TORCH_LR, verbose=False
    )
    sabr_seconds = time.perf_counter() - t0
    sparams = sabr_cal.result.params
    sabr_out = {
        "engine": "quantorch/MarketCalibrator (Adam, IV-space MSE, beta frozen at truth)",
        "start": {k: float(v) for k, v in sstart.items()},
        "recovered": {name: float(sparams[name]) for name in ("alpha", "beta", "rho", "nu")},
        "final_loss": float(sabr_cal.result.final_loss),
        "vol_rmse": float(sabr_cal.result.residual_summary["rmse"]),
        "num_epochs": TORCH_SABR_EPOCHS,
        "lr": TORCH_LR,
        "duration_s": sabr_seconds,
    }
    print(f"[quantorch] SABR fit done in {sabr_seconds:.1f}s: {sabr_out['recovered']}")

    payload = {"heston": heston_out, "sabr": sabr_out}
    with open(TMP_TORCH, "w") as fh:
        json.dump(payload, fh)
    print(f"[quantorch] wrote {TMP_TORCH}")


# ----------------------------------------------------------------------------
# Phase: calibrate with QuantLib (cross-check venv)
# ----------------------------------------------------------------------------


def _nelder_mead(f, x0, step=0.15, tol=1e-12, max_iter=4000):
    """Minimal Nelder-Mead (no numpy available in the QL venv)."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        vertex = list(x0)
        vertex[i] = vertex[i] * (1.0 + step) if vertex[i] != 0 else step
        simplex.append(vertex)
    vals = [f(v) for v in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if max(abs(simplex[k][j] - simplex[0][j]) for k in range(1, n + 1) for j in range(n)) < tol:
            break
        centroid = [sum(simplex[k][j] for k in range(n)) / n for j in range(n)]
        xr = [centroid[j] + 1.0 * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = [centroid[j] + 2.0 * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
            continue
        xc = [centroid[j] + 0.5 * (simplex[-1][j] - centroid[j]) for j in range(n)]
        fc = f(xc)
        if fc < vals[-1]:
            simplex[-1], vals[-1] = xc, fc
            continue
        for k in range(1, n + 1):
            simplex[k] = [simplex[0][j] + 0.5 * (simplex[k][j] - simplex[0][j]) for j in range(n)]
            vals[k] = f(simplex[k])
    best = min(range(n + 1), key=lambda i: vals[i])
    return simplex[best], vals[best]


def phase_qlfit() -> None:
    import QuantLib as ql

    surface = json.load(open(TMP_SURFACE))

    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    cal = ql.NullCalendar()
    dc = ql.Actual365Fixed()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(S0))
    rts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(RATE)), dc, ql.Continuous, ql.Annual)
    )
    qts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(DIV)), dc, ql.Continuous, ql.Annual)
    )

    # ---- Heston: HestonModel + HestonModelHelper + LevenbergMarquardt ----
    start = HESTON_START
    proc = ql.HestonProcess(rts, qts, spot_h, start["v0"], start["kappa"], start["theta"], start["sigma_v"], start["rho"])
    model = ql.HestonModel(proc)

    helpers = []
    for row in surface["heston"]:
        helper = ql.HestonModelHelper(
            ql.Period(row["days"], ql.Days),
            cal,
            S0,
            row["K"],
            ql.QuoteHandle(ql.SimpleQuote(row["iv"])),
            rts,
            qts,
            ql.BlackCalibrationHelper.RelativePriceError,
        )
        helpers.append(helper)

    # QuantLib-Python HestonModelHelper needs an explicit engine before calibrate()
    for helper in helpers:
        helper.setPricingEngine(ql.AnalyticHestonEngine(model, 64))

    method = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    end_criteria = ql.EndCriteria(2000, 1000, 1e-10, 1e-10, 1e-10)
    t0 = time.perf_counter()
    model.calibrate(helpers, method, end_criteria)
    ql_seconds = time.perf_counter() - t0
    end_type = model.endCriteria()

    recovered = {
        "kappa": model.kappa(),
        "theta": model.theta(),
        "sigma_v": model.sigma(),
        "rho": model.rho(),
        "v0": model.v0(),
    }

    # Post-fit price residuals (re-price with the calibrated model).
    engine = ql.AnalyticHestonEngine(model, 144)
    sq = 0.0
    mx = 0.0
    for row in surface["heston"]:
        mat = cal.advance(today, ql.Period(row["days"], ql.Days))
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, row["K"]), ql.EuropeanExercise(mat))
        opt.setPricingEngine(engine)
        diff = opt.NPV() - row["price"]
        sq += diff * diff
        mx = max(mx, abs(diff))

    heston_out = {
        "engine": "QuantLib HestonModel + HestonModelHelper + LevenbergMarquardt (RelativePriceError)",
        "start": {k: float(v) for k, v in start.items()},
        "recovered": {k: float(v) for k, v in recovered.items()},
        "end_criteria": str(end_type),
        "price_rmse": math.sqrt(sq / len(surface["heston"])),
        "price_max_abs_error": mx,
        "duration_s": ql_seconds,
    }
    print(f"[qlfit] Heston fit done in {ql_seconds:.2f}s: {heston_out['recovered']}")
    print(f"[qlfit] endCriteria={end_type} price_rmse={heston_out['price_rmse']:.4e}")

    # ---- SABR: ql.sabrVolatility residuals + Nelder-Mead (beta frozen) ----
    sabr_rows = surface["sabr"]
    sstart = SABR_START

    def objective(vec):
        alpha, rho, nu = vec
        if alpha <= 1e-6 or nu <= 1e-6 or abs(rho) >= 0.999:
            return 1e10
        s = 0.0
        for row in sabr_rows:
            model_vol = ql.sabrVolatility(row["K"], FWD, row["T"], alpha, SABR_TRUTH["beta"], nu, rho)
            s += (model_vol - row["vol"]) ** 2
        return s / len(sabr_rows)

    t0 = time.perf_counter()
    best_x, best_val = _nelder_mead(objective, [sstart["alpha"], sstart["rho"], sstart["nu"]], step=0.2)
    sabr_seconds = time.perf_counter() - t0
    sabr_out = {
        "engine": "ql.sabrVolatility (Hagan) residuals + Nelder-Mead (beta frozen at truth)",
        "start": {k: float(v) for k, v in sstart.items()},
        "recovered": {
            "alpha": float(best_x[0]),
            "beta": float(SABR_TRUTH["beta"]),
            "rho": float(best_x[1]),
            "nu": float(best_x[2]),
        },
        "final_loss": float(best_val),
        "vol_rmse": math.sqrt(best_val),
        "duration_s": sabr_seconds,
    }
    print(f"[qlfit] SABR fit done in {sabr_seconds:.2f}s: {sabr_out['recovered']}")

    payload = {"heston": heston_out, "sabr": sabr_out}
    with open(TMP_QL, "w") as fh:
        json.dump(payload, fh)
    print(f"[qlfit] wrote {TMP_QL}")


# ----------------------------------------------------------------------------
# Phase: report
# ----------------------------------------------------------------------------


def phase_report() -> None:
    surface = json.load(open(TMP_SURFACE))
    torch_out = json.load(open(TMP_TORCH))
    ql_out = json.load(open(TMP_QL))

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "setup": {
            "generator": "QuantLib 1.44 AnalyticHestonEngine (Laguerre order 144); ql.sabrVolatility for SABR",
            "heston_truth": HESTON_TRUTH,
            "heston_start": HESTON_START,
            "market": {"S0": S0, "r": RATE, "q": DIV, "strikes": STRIKES, "maturities_days": MAT_DAYS},
            "sabr_truth": SABR_TRUTH,
            "sabr_start": SABR_START,
            "notes": [],
        },
    }

    # ---- Heston section ----
    truth = HESTON_TRUTH
    qt = torch_out["heston"]
    qlh = ql_out["heston"]
    heston_section = {
        "truth": truth,
        "quantorch_recovered": qt["recovered"],
        "quantorch_start": qt["start"],
        "quantorch_rel_errors": {k: rel_err(qt["recovered"][k], truth[k]) for k in truth},
        "quantorch_price_rmse": qt["price_rmse"],
        "quantorch_final_loss": qt["final_loss"],
        "quantorch_duration_s": qt["duration_s"],
        "ql_recovered": qlh["recovered"],
        "ql_start": qlh["start"],
        "ql_rel_errors": {k: rel_err(qlh["recovered"][k], truth[k]) for k in truth},
        "ql_price_rmse": qlh["price_rmse"],
        "ql_end_criteria": qlh["end_criteria"],
        "ql_duration_s": qlh["duration_s"],
        "quantorch_vs_ql_abs_diff": {
            k: abs(qt["recovered"][k] - qlh["recovered"][k]) for k in truth
        },
    }
    results["heston"] = heston_section

    # ---- SABR section ----
    truth_s = {k: SABR_TRUTH[k] for k in ("alpha", "beta", "rho", "nu")}
    qts = torch_out["sabr"]
    qls = ql_out["sabr"]
    sabr_section = {
        "truth": truth_s,
        "quantorch_recovered": qts["recovered"],
        "quantorch_start": qts["start"],
        "quantorch_rel_errors": {k: rel_err(qts["recovered"][k], truth_s[k]) for k in truth_s},
        "quantorch_vol_rmse": qts["vol_rmse"],
        "quantorch_duration_s": qts["duration_s"],
        "ql_recovered": qls["recovered"],
        "ql_start": qls["start"],
        "ql_rel_errors": {k: rel_err(qls["recovered"][k], truth_s[k]) for k in truth_s},
        "ql_vol_rmse": qls["vol_rmse"],
        "ql_duration_s": qls["duration_s"],
        "quantorch_vs_ql_abs_diff": {
            k: abs(qts["recovered"][k] - qls["recovered"][k]) for k in truth_s
        },
    }
    results["sabr"] = sabr_section

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- readable tables ----
    def fmt_row(name, truth_v, qt_v, ql_v):
        qe = rel_err(qt_v, truth_v)
        le = rel_err(ql_v, truth_v)
        qe_s = f"{qe:.2e}" + ("!" if qe > 1e-2 else "")
        le_s = f"{le:.2e}" + ("!" if le > 1e-2 else "")
        return f"  {name:<10} {truth_v:>10.5f} {qt_v:>14.6f} {qe_s:>12} {ql_v:>14.6f} {le_s:>12}"

    print("=" * 84)
    print("Heston calibration recovery (surface: 5 strikes x 4 maturities, QL-generated)")
    print(f"  start point: {HESTON_START}")
    print(f"  {'param':<10} {'truth':>10} {'quantorch':>14} {'rel_err':>12} {'QuantLib':>14} {'rel_err':>12}")
    for k in ("kappa", "theta", "sigma_v", "rho", "v0"):
        print(fmt_row(k, HESTON_TRUTH[k], qt["recovered"][k], qlh["recovered"][k]))
    print(f"  price RMSE:  quantorch={qt['price_rmse']:.3e}  QuantLib={qlh['price_rmse']:.3e}")
    print(f"  (! = rel_err > 1e-2 target tolerance)")

    print()
    print("=" * 84)
    print("SABR calibration recovery (Hagan lognormal vols, beta frozen at truth=0.5)")
    print(f"  start point (free params): {{'alpha': 1.5, 'rho': -0.10, 'nu': 0.90}}")
    print(f"  {'param':<10} {'truth':>10} {'quantorch':>14} {'rel_err':>12} {'QuantLib/NM':>14} {'rel_err':>12}")
    for k in ("alpha", "beta", "rho", "nu"):
        print(fmt_row(k, SABR_TRUTH[k], qts["recovered"][k], qls["recovered"][k]))
    print(f"  vol RMSE:    quantorch={qts['vol_rmse']:.3e}  QL/NM={qls['vol_rmse']:.3e}")
    print(f"\nwrote {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["generate", "quantorch", "qlfit", "report"])
    args = parser.parse_args()
    {"generate": phase_generate, "quantorch": phase_quantorch, "qlfit": phase_qlfit, "report": phase_report}[
        args.phase
    ]()


if __name__ == "__main__":
    main()
