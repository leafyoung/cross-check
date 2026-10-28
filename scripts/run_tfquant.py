#!/usr/bin/env python
"""Run the canonical cross-check cases through tf-quant-finance (JAX).
Run from the tf-quant-finance repo dir so `uv run` picks up its env:

  cd /var/home/yangye/devv/qfin/pricing/tf-quant-finance
  XLA_PYTHON_CLIENT_PREALLOCATE=false nice -n 19 \
    uv run python ../cross-check/scripts/run_tfquant.py

Writes results/tfquant-<platform>.json (cross-check/results/). Never imports
tf_quant_finance.experimental.pricing_platform (hipSparse crash — repo rule).
tf-quant scope in the catalog: BSM vanilla, CRR binomial, implied vol,
analytic Asian/barrier/binary, PDE European, swap-curve bootstrap, and the
six tfqr_ rates families (Hagan-West bootstrap + monotone convex, NSS fit via
tff's BFGS over its NSS interpolation, forwards/cashflow identities).
Everything else is marked skipped (no tf-quant equivalent in scope: no
Bermudan tree, no Heston analytic, no SABR Hagan IV, no one-factor ZCB closed
forms, no MC exotics/risk modules).
"""
import json
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from tf_quant_finance.black_scholes import option_price, implied_vol, implied_vol_approx, asian_option_price
from tf_quant_finance.black_scholes.crr_binomial_tree import option_price_binomial
from tf_quant_finance.black_scholes.vanilla_prices import barrier_price, binary_price
from tf_quant_finance.math.pde import fd_solvers, grids, boundary_conditions
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math.optimizer import bfgs_minimize
from tf_quant_finance.rates.hagan_west import bond_curve, monotone_convex
from tf_quant_finance.rates.nelson_seigel_svensson import interpolate as nss_interpolate
from tf_quant_finance.rates.nelson_seigel_svensson import SvenssonParameters
from tf_quant_finance.rates.analytics import cashflows
from tf_quant_finance.rates.analytics import forwards as fwd_analytics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SUPPORTED = {
    "bsm_eu_call", "bsm_eu_put", "crr_eu_call", "crr_amer_put", "amer_put_100",
    "tfq_iv_newton_call", "tfq_iv_newton_put", "tfq_iv_approx_atm",
    "tfq_asian_geo_m", "tfq_asian_arith_lb",
    "tfq_barrier_doko_call", "tfq_barrier_uoko_put",
    "tfq_barrier_diki_call", "tfq_barrier_uiki_put",
    "tfq_binary_call", "tfq_binary_put", "tfq_pde_eu_call", "tfq_swapcurve_flat",
    "tfqr_hw_bondcurve", "tfqr_hw_monotone", "tfqr_hw_zcb",
    "tfqr_nss_fit", "tfqr_forwards", "tfqr_cashflows",
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


def price_hw_bondcurve(p):
    """tff Hagan-West bond-curve bootstrap (monotone convex basis)."""
    res = bond_curve(
        [np.array(c, dtype=np.float64) for c in p["bond_cashflows"]],
        [np.array(t, dtype=np.float64) for t in p["bond_cashflow_times"]],
        np.array(p["present_values"], dtype=np.float64), dtype=np.float64)
    return {"rates": [float(r) for r in res.discount_rates],
            "dfs": [float(d) for d in res.discount_factors]}


def price_hw_monotone(p):
    """tff monotone-convex interpolation: forwards (plain + negative-forward
    stress curve) and yields off the paper yield curve."""
    yc = p["yields_curve"]
    y_vals = monotone_convex.interpolate_yields(
        jnp.array(yc["query_times"]), jnp.array(yc["reference_times"]),
        yields=jnp.array(yc["yields"]), dtype=jnp.float64)
    fc = p["fwd_curve"]
    f_vals = monotone_convex.interpolate_forward_rate(
        jnp.array(fc["query_times"]), jnp.array(fc["reference_times"]),
        discrete_forwards=jnp.array(fc["forwards"]), dtype=jnp.float64)
    fn = p["fwd_curve_neg"]
    fn_vals = monotone_convex.interpolate_forward_rate(
        jnp.array(fn["query_times"]), jnp.array(fn["reference_times"]),
        discrete_forwards=jnp.array(fn["forwards"]), dtype=jnp.float64)
    # yields from discrete forwards (internally converts forwards->yields)
    yn_vals = monotone_convex.interpolate_yields(
        jnp.array(fn["query_times"]), jnp.array(fn["reference_times"]),
        discrete_forwards=jnp.array(fn["forwards"]), dtype=jnp.float64)
    return {"forwards": [float(v) for v in f_vals],
            "yields": [float(v) for v in y_vals],
            "fwd_neg": [float(v) for v in fn_vals],
            "yld_neg": [float(v) for v in yn_vals]}


def price_hw_zcb(p):
    """tff Hagan-West bootstrap on a zero-coupon strip: node DFs are pinned
    by the instrument prices (no interpolation freedom at nodes)."""
    prm = p["nss_params"]

    def nss(t):
        h1 = (1 - jnp.exp(-t / prm["tau1"])) / (t / prm["tau1"])
        h2 = (1 - jnp.exp(-t / prm["tau2"])) / (t / prm["tau2"])
        return (prm["beta0"] + prm["beta1"] * h1 +
                prm["beta2"] * (h1 - jnp.exp(-t / prm["tau1"])) +
                prm["beta3"] * (h2 - jnp.exp(-t / prm["tau2"])))

    ts = [d / 365.0 for d in p["zcb_days"]]
    dfs_true = [float(jnp.exp(-nss(_t64(t)) * t)) for t in ts]
    res = bond_curve(
        [np.array([p["face"]]) for _ in ts],
        [np.array([t]) for t in ts],
        np.array([p["face"] * d for d in dfs_true], dtype=np.float64),
        dtype=np.float64)
    return {"dfs": [float(d) for d in res.discount_factors]}


def price_nss_fit(p):
    """Fit NSS parameters to synthetic yields with tff's BFGS over tff's
    NSS interpolation (tff ships the parametric form; the least-squares fit
    is harness-side). taus are fitted in log space to keep them positive.
    Compare fitted CURVE VALUES at the test maturities (fit tenors), not the
    (non-unique) parameters themselves."""
    prm = p["true_params"]

    def nss_py(t):
        h1 = (1 - jnp.exp(-t / prm["tau1"])) / (t / prm["tau1"])
        h2 = (1 - jnp.exp(-t / prm["tau2"])) / (t / prm["tau2"])
        return (prm["beta0"] + prm["beta1"] * h1 +
                prm["beta2"] * (h1 - jnp.exp(-t / prm["tau1"])) +
                prm["beta3"] * (h2 - jnp.exp(-t / prm["tau2"])))

    tt = jnp.array(p["fit_tenors"])
    yy = jnp.array([float(nss_py(t)) for t in p["fit_tenors"]])

    @make_val_and_grad_fn
    def loss(u):
        sp = SvenssonParameters(beta_0=u[0], beta_1=u[1], beta_2=u[2],
                                beta_3=u[3], tau_1=jnp.exp(u[4]),
                                tau_2=jnp.exp(u[5]))
        return jnp.mean((nss_interpolate(tt, sp) - yy) ** 2)

    res = bfgs_minimize(loss, jnp.array(p["fit_start"], dtype=jnp.float64),
                        tolerance=1e-12, max_iterations=500)
    u = res.position
    sp = SvenssonParameters(beta_0=u[0], beta_1=u[1], beta_2=u[2], beta_3=u[3],
                            tau_1=jnp.exp(u[4]), tau_2=jnp.exp(u[5]))
    test_ts = [p["fit_tenors"][i] for i in p["test_indices"]]
    fitted = nss_interpolate(jnp.array(test_ts), sp)
    return {"yields": [float(v) for v in fitted]}


def price_fwd_identities(p):
    """forwards.from_yields / to_yields / from_discount_factors identities."""
    prm = p["nss_params"]

    def nss(t):
        h1 = (1 - jnp.exp(-t / prm["tau1"])) / (t / prm["tau1"])
        h2 = (1 - jnp.exp(-t / prm["tau2"])) / (t / prm["tau2"])
        return (prm["beta0"] + prm["beta1"] * h1 +
                prm["beta2"] * (h1 - jnp.exp(-t / prm["tau1"])) +
                prm["beta3"] * (h2 - jnp.exp(-t / prm["tau2"])))

    ts = jnp.array(p["times"])
    ys = nss(ts)
    fwds = fwd_analytics.forward_rates_from_yields(ys, ts, dtype=jnp.float64)
    back = fwd_analytics.yields_from_forward_rates(fwds, ts, dtype=jnp.float64)
    dfs = jnp.exp(-ys * ts)
    df_fwd = fwd_analytics.forward_rates(
        dfs[:-1], dfs[1:], ts[1:] - ts[:-1], dtype=jnp.float64)
    return {"fwd": [float(v) for v in fwds],
            "roundtrip": [float(v) for v in back],
            "df_fwd": [float(v) for v in df_fwd]}


def price_cashflow_identities(p):
    """rates.analytics.cashflows: PV at flat yield, Newton yield recovery,
    PV from explicit discount factors."""
    ts = jnp.array(p["cashflow_times"])
    cfs = jnp.array([p["coupon"]] * (len(ts) - 1) + [p["final_cashflow"]])
    y = _t64(p["flat_yield"])
    pv = cashflows.pv_from_yields(cfs, ts, jnp.array([y]), dtype=jnp.float64)
    y_rec = cashflows.yields_from_pv(cfs, ts, pv, dtype=jnp.float64)
    dfs = jnp.exp(-y * ts)
    pv2 = cashflows.present_value(cfs[None, :], dfs[None, :],
                                  dtype=jnp.float64)
    return {"pv": [float(pv[0])], "yield": [float(y_rec[0])],
            "pv_dfs": [float(pv2[0])]}


MODEL_DISPATCH = {
    "iv": price_iv,
    "asian_analytic": price_asian_analytic,
    "barrier": price_barrier,
    "binary": price_binary,
    "pde_eu": price_pde_eu,
    "swap_curve": price_swap_curve,
    "hw_bondcurve": price_hw_bondcurve,
    "hw_monotone": price_hw_monotone,
    "hw_zcb": price_hw_zcb,
    "nss_fit": price_nss_fit,
    "fwd_id": price_fwd_identities,
    "cf_id": price_cashflow_identities,
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
    priced = sum(1 for r in results if r["ok"] and
                 (r.get("price") is not None or r.get("values")))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = [r["id"] for r in results if not r["ok"]]
    print(f"tfquant[{backend}] (devices: {platforms}): {priced} priced, {skipped} skipped, "
          f"failures: {failed or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
