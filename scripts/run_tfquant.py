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

from tf_quant_finance.black_scholes import option_price, implied_vol, implied_vol_approx, asian_option_price
from tf_quant_finance.black_scholes.crr_binomial_tree import option_price_binomial
from tf_quant_finance.black_scholes.vanilla_prices import barrier_price, binary_price
from tf_quant_finance.math.pde import fd_solvers, grids, boundary_conditions

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SUPPORTED = {
    "bsm_eu_call", "bsm_eu_put", "crr_eu_call", "crr_amer_put", "amer_put_100",
    "tfq_iv_newton_call", "tfq_iv_newton_put", "tfq_iv_approx_atm",
    "tfq_asian_geo_m", "tfq_asian_arith_lb",
    "tfq_barrier_doko_call", "tfq_barrier_uoko_put",
    "tfq_barrier_diki_call", "tfq_barrier_uiki_put",
    "tfq_binary_call", "tfq_binary_put", "tfq_pde_eu_call", "tfq_swapcurve_flat",
}



def _t64(x):
    return jnp.array(x, dtype=jnp.float64)


def price_iv(p):
    """Implied-vol roundtrip: invert the QuantLib-generated market price."""
    if p.get("method") == "approx":
        vol = implied_vol_approx(
            prices=_t64(p["market_price"]), strikes=_t64(p["strike"]),
            expiries=_t64(p["expiry"]), forwards=_t64(
                p["spot"] * jnp.exp((p["rate"] - p["dividend"]) * p["expiry"])),
            discount_factors=_t64(jnp.exp(-p["rate"] * p["expiry"])),
            is_call_options=_t64(p["option_type"] == "call"))
        return float(vol)
    vol = implied_vol(
        prices=_t64(p["market_price"]), strikes=_t64(p["strike"]),
        expiries=_t64(p["expiry"]), forwards=_t64(
            p["spot"] * jnp.exp((p["rate"] - p["dividend"]) * p["expiry"])),
        discount_factors=_t64(jnp.exp(-p["rate"] * p["expiry"])),
        is_call_options=_t64(p["option_type"] == "call"), dtype=jnp.float64)
    return float(vol)


def price_asian_analytic(p):
    val = asian_option_price(
        volatilities=_t64(p["volatility"]), strikes=_t64(p["strike"]),
        expiries=_t64(p["expiry"]), spots=_t64(p["spot"]),
        sampling_times=jnp.array(p["sampling_times"], dtype=jnp.float64)[:, None],
        discount_rates=_t64(p["rate"]), dividend_rates=_t64(p["dividend"]),
        is_call_options=_t64(p["option_type"] == "call"), dtype=jnp.float64)
    return float(jnp.squeeze(val))


def price_barrier(p):
    val = barrier_price(
        volatilities=_t64(p["volatility"]), strikes=_t64(p["strike"]),
        expiries=_t64(p["expiry"]), spots=_t64(p["spot"]),
        barriers=_t64(p["barrier"]), rebates=_t64(0.0),
        discount_rates=_t64(p["rate"]), dividend_rates=_t64(p["dividend"]),
        is_barrier_down=_t64(p["barrier_type"].startswith("down")),
        is_knock_out=_t64("out" in p["barrier_type"]),
        is_call_options=_t64(p["option_type"] == "call"), dtype=jnp.float64)
    return float(val)


def price_binary(p):
    val = binary_price(
        volatilities=_t64(p["volatility"]), strikes=_t64(p["strike"]),
        expiries=_t64(p["expiry"]), spots=_t64(p["spot"]),
        discount_rates=_t64(p["rate"]), dividend_rates=_t64(p["dividend"]),
        is_call_options=_t64(p["option_type"] == "call"), dtype=jnp.float64)
    return float(val)


def price_pde_eu(p):
    """Vanilla European via tff's 1-D parabolic FD solver (docstring recipe)."""
    import jax.numpy as jnp
    s_min, s_max = 1e-5, float(p["s_max"])
    grid = grids.uniform_grid(minimums=[s_min], maximums=[s_max],
                              sizes=[int(p["num_grid_points"])], dtype=jnp.float64)
    strike = jnp.array([[p["strike"]]], dtype=jnp.float64)
    volatility = jnp.array([[p["volatility"]]], dtype=jnp.float64)
    rate = jnp.array([[p["rate"]]], dtype=jnp.float64)
    dividend = jnp.array([[p["dividend"]]], dtype=jnp.float64)
    expiry = float(p["expiry"])
    s = grid[0]
    final_value_grid = jnp.maximum(s - strike, 0.0)

    def second_order_coeff_fn(t, grid_):
        del t
        s_ = grid_[0]
        return [[volatility**2 * s_**2 / 2]]

    def first_order_coeff_fn(t, grid_):
        del t
        s_ = grid_[0]
        return [(rate - dividend) * s_]

    def zeroth_order_coeff_fn(t, grid_):
        del t, grid_
        return -rate

    @boundary_conditions.dirichlet
    def lower_boundary_fn(t, grid_):
        del t, grid_
        return 0.0

    @boundary_conditions.dirichlet
    def upper_boundary_fn(t, grid_):
        del grid_
        return jnp.squeeze(s_max - strike * jnp.exp(-rate * (expiry - t)))

    estimate = fd_solvers.solve_backward(
        start_time=expiry, end_time=0.0,
        coord_grid=grid, values_grid=final_value_grid,
        time_step=float(p["time_step"]),
        boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn,
        dtype=jnp.float64)[0]
    # Interpolate the grid at the option's spot.
    xs = estimate[0, :]
    ys = estimate[1, :] if estimate.ndim > 1 else xs
    vals = jnp.squeeze(estimate)
    # vals has shape [batch(1), grid]; take batch 0.
    vals = jnp.reshape(vals, (vals.shape[-1],)) if vals.ndim == 1 else vals[0]
    return float(jnp.interp(p["spot"], s, vals))


def price_swap_curve(p):
    """Bootstrap a zero curve from par swaps; output discount factors at the
    test maturities (dict result, compared against golden dfs)."""
    import math
    r_flat = float(p["flat_rate"])
    ts = [float(t) for t in p["swap_maturities"]]
    import numpy as np
    float_leg_start_times, float_leg_end_times = [], []
    fixed_leg_start_times, fixed_leg_end_times = [], []
    fixed_leg_cashflows, present_values = [], []
    n_floats = []
    for T in ts:
        n_f = int(round(T / 0.5))
        n_floats.append(n_f)
        float_leg_start_times.append(np.array([0.5 * k for k in range(n_f)]))
        float_leg_end_times.append(np.array([0.5 * (k + 1) for k in range(n_f)]))
        fixed_leg_start_times.append(np.array([float(k) for k in range(int(round(T)))]))
        fixed_leg_end_times.append(np.array([float(k + 1) for k in range(int(round(T)))]))
        # par rate under the flat curve (same recipe as the golden side)
        float_leg = sum(0.5 * (math.exp(r_flat * 0.5) - 1.0) * math.exp(-r_flat * 0.5 * k)
                        for k in range(1, n_f + 1))
        annuity = sum(math.exp(-r_flat * k) for k in range(1, int(round(T)) + 1))
        par = float_leg / annuity
        # Library convention: fixed-leg cashflows are NEGATIVE (fixed leg
        # paid) — see swap_curve_test.py::test_correctness_bootstrap.
        fixed_leg_cashflows.append(np.array([-par] * int(round(T))))
        present_values.append(np.array(0.0))  # par swap
    from tf_quant_finance.rates.swap_curve_bootstrap import swap_curve_bootstrap
    built = swap_curve_bootstrap(
        float_leg_start_times=float_leg_start_times,
        float_leg_end_times=float_leg_end_times,
        float_leg_daycount_fractions=[np.full(n_f, 0.5) for n_f in n_floats],
        fixed_leg_start_times=fixed_leg_start_times,
        fixed_leg_end_times=fixed_leg_end_times,
        fixed_leg_cashflows=fixed_leg_cashflows,
        fixed_leg_daycount_fractions=[np.ones(int(round(T))) for T in ts],
        present_values=[_t64(v) for v in present_values],
        initial_curve_rates=jnp.full([len(ts)], r_flat, dtype=jnp.float64),
        curve_interpolator=lambda xi, x, y: jnp.interp(xi, x, y),
        dtype=jnp.float64)
    # built.zero_rates at swap end times; discount = exp(-r * t)
    rates = jnp.squeeze(built.rates)
    node_times = jnp.array(ts, dtype=jnp.float64)
    dfs = [float(jnp.exp(-jnp.interp(t, node_times, rates) * t)) for t in p["test_maturities"]]
    return {"dfs": dfs}


MODEL_DISPATCH = {
    "iv": price_iv,
    "asian_analytic": price_asian_analytic,
    "barrier": price_barrier,
    "binary": price_binary,
    "pde_eu": price_pde_eu,
    "swap_curve": price_swap_curve,
}

def price_case(model, p):
    if model in MODEL_DISPATCH:
        return MODEL_DISPATCH[model](p)
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
                out = price_case(c["model"], c["params"])
                if isinstance(out, dict):
                    entry.update(ok=True, values=out)
                else:
                    entry.update(ok=True, price=float(out))
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
