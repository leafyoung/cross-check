#!/usr/bin/env python
"""Run the canonical cross-check cases through torchquantlib (quantorch).
Usage: quantorch/.venv/bin/python scripts/run_quantorch.py   (from cross-check/)
Writes results/quantorch-cpu.json in the standard result shape.

Backend note: quantorch's uv.lock pins plain PyPI torch (CUDA build) — there is
no ROCm wheel in its lockfile, and this box has no NVIDIA GPU, so execution is
CPU regardless. We install the CPU-only torch wheel to avoid the CUDA deps.
Stochastic cases use torch's own RNG (manual_seed + randn); draws cannot match
quantburn/quantcandle's Xorshift64Star stream, so those results carry
"rng_divergent": true and compare only at Monte-Carlo tolerance.
"""
import json
import sys
from pathlib import Path

import torch

from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton
from torchquantlib.core.asset_pricing.option.american_option import american_option
from torchquantlib.core.asset_pricing.option.bermudan_option import bermudan_option
from torchquantlib.core.asset_pricing.option.asian_option import asian_option
from torchquantlib.core.asset_pricing.option.exotics import (
    barrier_option,
    chooser_option,
    digital_option,
    lookback_option,
)
from torchquantlib.models.binomial_tree import binomial_tree
from torchquantlib.models.interest_rate.black_karasinski import BlackKarasinski
from torchquantlib.models.interest_rate.cir import CIR
from torchquantlib.models.interest_rate.hull_white import HullWhiteModel as HullWhite
from torchquantlib.models.interest_rate.vasicek import Vasicek
from torchquantlib.models.stochastic_volatility.heston import Heston
from torchquantlib.models.stochastic_volatility.sabr import SABR, _black76_price
from torchquantlib.core.risk.market_risk.expected_shortfall import calculate_es
from torchquantlib.core.risk.market_risk.var import calculate_var
from torchquantlib.core.risk.greeks.greeks import MalliavinGreeks

ROOT = Path(__file__).resolve().parent.parent

# Models whose result depends on torch's own RNG stream (divergent from the
# Xorshift64Star stream used by quantburn/quantcandle).
STOCHASTIC = {"asian_call", "heston_mc_call", "digital_call", "barrier_uo", "lookback_call"}


def t64(value):
    return torch.as_tensor(value, dtype=torch.float64)


def price_case(model, p):
    if model == "bsm":
        return float(black_scholes_merton(
            p["option_type"], p["option_style"],
            t64(p["spot"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]), t64(p["dividend"]),
        ).item())
    if model == "binomial":
        return float(binomial_tree(
            p["option_type"], p["option_style"],
            t64([p["spot"]]), t64([p["strike"]]), t64([p["expiry"]]),
            t64([p["volatility"]]), t64([p["rate"]]), p["steps"],
        ).item())
    if model == "american_option":
        return float(american_option(
            p["option_type"], t64(p["spot"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]), p["steps"],
        ).item())
    if model == "bermudan_option":
        return float(bermudan_option(
            p["option_type"], t64(p["spot"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]), p["steps"],
            t64(p["exercise_dates"]).to(torch.int64),
        ).item())
    if model == "asian_option":
        torch.manual_seed(p.get("seed", 2007))
        return float(asian_option(
            p["option_type"], t64(p["spot"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]), p["steps"],
            average_type=p.get("average_type", "arithmetic"),
            num_paths=p.get("num_paths"), seed=p.get("seed", 2007),
        ).item())
    if model == "heston_cf":
        h = Heston(kappa_init=p["kappa"], theta_init=p["theta"], sigma_v_init=p["sigma_v"],
                   rho_init=p["rho"], v0_init=p["v0"], mu_init=p.get("mu", 0.0))
        return float(h.characteristic_function_price(
            p["spot"], p["strike"], p["expiry"], p["rate"],
            option_type=p["option_type"], q=p["dividend"],
            integration_limit=p.get("integration_limit", 100.0),
            integration_points=p.get("integration_points", 2000),
        ).item())
    if model == "heston_mc":
        h = Heston(kappa_init=p["kappa"], theta_init=p["theta"], sigma_v_init=p["sigma_v"],
                   rho_init=p["rho"], v0_init=p["v0"], mu_init=p.get("mu", 0.0))
        return float(h.monte_carlo_option_price(
            p["spot"], p["strike"], p["expiry"], p["rate"],
            option_type=p["option_type"], q=p["dividend"],
            N=p["n_paths"], steps=p["steps"], seed=p.get("seed", 2007),
        ).item())
    if model == "sabr_iv":
        s = SABR(alpha_init=p["alpha"], beta_init=p["beta"], rho_init=p["rho"],
                 nu_init=p["nu"], F0=p["f0"], shift_init=p.get("shift", 0.0))
        return float(s.implied_volatility(p["strike"], p["expiry"]).item())
    if model == "sabr_price":
        s = SABR(alpha_init=p["alpha"], beta_init=p["beta"], rho_init=p["rho"],
                 nu_init=p["nu"], F0=p["f0"], shift_init=p.get("shift", 0.0))
        return float(s.option_price(
            p["strike"], p["expiry"], p["rate"], option_type=p["option_type"],
            model=p.get("model", "lognormal"), shift=p.get("shift"),
        ))
    if model == "black76":
        # No public Black-76 in torchquantlib; the SABR module's private helper
        # is the library's own implementation.
        return float(_black76_price(
            p["option_type"], t64(p["forward"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]),
        ).item())
    if model == "digital_option":
        torch.manual_seed(p.get("seed", 2007))
        return float(digital_option(
            p["option_type"], t64(p["spot"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]), t64(p["payout"]), p["n_paths"],
        ).item())
    if model == "barrier_option":
        torch.manual_seed(p.get("seed", 2007))
        return float(barrier_option(
            p["option_type"], p["barrier_type"], t64(p["spot"]), t64(p["strike"]),
            t64(p["barrier"]), t64(p["expiry"]), t64(p["volatility"]), t64(p["rate"]),
            p["n_paths"], p["steps"],
        ).item())
    if model == "lookback_option":
        torch.manual_seed(p.get("seed", 2007))
        return float(lookback_option(
            p["option_type"], t64(p["spot"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]), p["n_paths"],
            strike_type=p.get("strike_type", "fixed"),
        ).item())
    if model == "chooser_option":
        return float(chooser_option(
            t64(p["spot"]), t64(p["strike"]), t64(p["expiry"]),
            t64(p["volatility"]), t64(p["rate"]), t64(p["choose_time"]),
        ).item())
    if model == "vasicek_zcb":
        v = Vasicek(kappa_init=p["kappa"], theta_init=p["theta"],
                    sigma_init=p["sigma"], r0_init=p["r0"])
        return float(v.zero_coupon_bond_price(p["maturity"]).item())
    if model == "cir_zcb":
        c = CIR(kappa_init=p["kappa"], theta_init=p["theta"],
                sigma_init=p["sigma"], r0_init=p["r0"])
        return float(c.zero_coupon_bond_price(p["maturity"]).item())
    if model == "hull_white_zcb":
        h = HullWhite(a_init=p["a"], sigma_init=p["sigma"], r0_init=p["r0"])
        return float(h.zero_coupon_bond_price(p["maturity"]).item())
    if model == "black_karasinski_zcb":
        b = BlackKarasinski(a_init=p["a"], sigma_init=p["sigma"], r0_init=p["r0"])
        return float(b.zero_coupon_bond_price(p["maturity"], None, 100).item())
    if model == "greeks_european":
        g = MalliavinGreeks().european_option_greeks(
            p["spot"], p["strike"], p["expiry"], p["rate"], p["volatility"],
            num_paths=p.get("n_paths", 100000), seed=p.get("seed", 42))
        return {"delta": g["Delta"], "gamma": g["Gamma"], "vega": g["Vega"],
                "theta": g["Theta"], "rho": g["Rho"]}
    if model == "var":
        return float(calculate_var(t64(p["returns"]), p["confidence_level"]).item())
    if model == "es":
        return float(calculate_es(t64(p["returns"]), p["confidence_level"]).item())
    raise ValueError(f"unknown model '{model}'")


def main():
    cases = json.loads((ROOT / "cases" / "cases.json").read_text())["cases"]
    results = []
    for c in cases:
        entry = {"id": c["id"], "ok": False}
        try:
            out = price_case(c["model"], c["params"])
            if isinstance(out, dict):
                entry.update(ok=True, values={k: float(v) for k, v in out.items()})
            else:
                entry.update(ok=True, price=float(out))
            if c["id"] in STOCHASTIC:
                entry["rng_divergent"] = True
        except Exception as e:
            entry.update(ok=False, error=f"{type(e).__name__}: {e}")
        results.append(entry)
    out = {"backend": "cpu", "results": results}
    (ROOT / "results" / "quantorch-cpu.json").write_text(json.dumps(out, indent=2))
    ok = sum(1 for r in results if r["ok"])
    skipped = [r["id"] for r in results if not r["ok"]]
    print(f"quantorch[cpu]: {ok}/{len(results)} ok; failures: {skipped or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
