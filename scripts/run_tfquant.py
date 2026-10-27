#!/usr/bin/env python
"""Run the canonical cross-check cases through tf-quant-finance (JAX).
Run from the tf-quant-finance repo dir so `uv run` picks up its env:

  cd /var/home/yangye/devv/qfin/pricing/tf-quant-finance
  XLA_PYTHON_CLIENT_PREALLOCATE=false nice -n 19 \
    uv run python ../cross-check/scripts/run_tfquant.py

Writes results/tfquant-<platform>.json (cross-check/results/). Never imports
tf_quant_finance.experimental.pricing_platform (hipSparse crash — repo rule).
tf-quant overlap with the case catalog: BSM vanilla (European) and CRR binomial
(European/American). Everything else is marked skipped (no tf-quant equivalent
in scope: no Bermudan tree, no Heston analytic, no SABR Hagan IV, no ZCB closed
forms, no exotics/risk modules).
"""
import json
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tf_quant_finance.black_scholes import option_price
from tf_quant_finance.black_scholes.crr_binomial_tree import option_price_binomial

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SUPPORTED = {"bsm_eu_call", "bsm_eu_put", "crr_eu_call", "crr_amer_put", "amer_put_100"}


def price_case(model, p):
    if model == "bsm":
        return float(option_price(
            volatilities=jnp.array(p["volatility"], dtype=jnp.float64),
            strikes=jnp.array(p["strike"], dtype=jnp.float64),
            expiries=jnp.array(p["expiry"], dtype=jnp.float64),
            spots=jnp.array(p["spot"], dtype=jnp.float64),
            discount_rates=jnp.array(p["rate"], dtype=jnp.float64),
            dividend_rates=jnp.array(p["dividend"], dtype=jnp.float64),
            is_call_options=jnp.array(p["option_type"] == "call"),
            dtype=jnp.float64,
        ))
    if model in ("binomial", "american_option"):
        is_american = p.get("option_style", "american") == "american" or model == "american_option"
        return float(option_price_binomial(
            volatilities=jnp.array(p["volatility"], dtype=jnp.float64),
            strikes=jnp.array(p["strike"], dtype=jnp.float64),
            expiries=jnp.array(p["expiry"], dtype=jnp.float64),
            spots=jnp.array(p["spot"], dtype=jnp.float64),
            discount_rates=jnp.array(p["rate"], dtype=jnp.float64),
            dividend_rates=jnp.array(0.0, dtype=jnp.float64),
            is_call_options=jnp.array(p["option_type"] == "call"),
            is_american=jnp.array(is_american),
            num_steps=p["steps"],
            dtype=jnp.float64,
        ))
    raise ValueError(f"no tf-quant mapping for model '{model}'")


def main():
    cases = json.loads((ROOT / "cases" / "cases.json").read_text())["cases"]
    results = []
    for c in cases:
        entry = {"id": c["id"], "ok": False}
        if c["id"] not in SUPPORTED:
            entry.update(ok=True, price=None, skipped=True, note="no tf-quant equivalent in scope")
        else:
            try:
                entry.update(ok=True, price=float(price_case(c["model"], c["params"])))
            except Exception as e:
                entry.update(ok=False, error=f"{type(e).__name__}: {e}")
        results.append(entry)

    platforms = [d.platform for d in jax.devices()]
    backend = platforms[0] if platforms else "cpu"
    out = {"backend": backend, "results": results}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"tfquant-{backend}.json").write_text(json.dumps(out, indent=2))
    priced = sum(1 for r in results if r["ok"] and r.get("price") is not None)
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = [r["id"] for r in results if not r["ok"]]
    print(f"tfquant[{backend}] (devices: {platforms}): {priced} priced, {skipped} skipped, "
          f"failures: {failed or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
