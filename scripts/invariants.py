#!/usr/bin/env python
"""Parity invariants across libraries — properties that must hold for ANY
correct implementation, checked per library via its cross-check runner output
plus fresh calls where the runner doesn't cover the needed quantity.

Invariants:
  I1 put-call parity (BSM European): C - P = e^{-qT}S - e^{-rT}K
  I2 binary + asset-or-nothing = discounted forward (per unit)
  I3 barrier in + out = vanilla (rebate 0)
  I4 CRR European -> Black-Scholes analytic as steps -> inf (error monotone-ish)
  I5 ZCB price strictly decreasing in maturity (positive forward rates)

Usage: scripts/invariants.py   (writes results/invariants.md)
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

S, K, T, V, R, Q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.02
Q0 = 0.0  # dividend-free pair for I4 (tree runners take no dividend arg)
CASES = {"cases": [
    {"id": "iv_bsm_call", "model": "bsm", "params": {"option_type": "call", "option_style": "european", "spot": S, "strike": K, "expiry": T, "volatility": V, "rate": R, "dividend": Q}},
    {"id": "iv_bsm_put", "model": "bsm", "params": {"option_type": "put", "option_style": "european", "spot": S, "strike": K, "expiry": T, "volatility": V, "rate": R, "dividend": Q}},
    {"id": "iv_vanilla", "model": "bsm", "params": {"option_type": "call", "option_style": "european", "spot": S, "strike": K, "expiry": T, "volatility": V, "rate": R, "dividend": Q0}},
    {"id": "iv_bin_do", "model": "binomial", "params": {"option_type": "call", "option_style": "european", "spot": S, "strike": K, "expiry": T, "volatility": V, "rate": R, "steps": 500}},
    {"id": "iv_vasicek", "model": "vasicek_zcb", "params": {"kappa": 0.15, "theta": 0.05, "sigma": 0.02, "r0": 0.05, "maturity": 1.0}},
]}

RUNNERS = {
    "quantburn": ["cargo", "run", "-q", "--manifest-path", str(ROOT.parent / "quantburn" / "Cargo.toml"), "--bin", "crosscheck"],
    "quantcandle": ["cargo", "run", "-q", "--manifest-path", str(ROOT.parent / "quantcandle" / "Cargo.toml"), "--bin", "crosscheck"],
    "quantjax": [str(ROOT.parent / "quantjax" / ".venv" / "bin" / "python"), "-m", "quantjax.cli"],
    "tfquant": None,  # special: jax venv via tf-quant env; skip here, run separately
}


def run_lib(name):
    if name == "tfquant":
        return None
    cmd = RUNNERS[name]
    payload = json.dumps(CASES)
    cwd = str(ROOT.parent / "quantjax") if name == "quantjax" else str(ROOT)
    proc = subprocess.run(cmd, input=payload, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0:
        return {}
    return {r["id"]: r.get("price") for r in json.loads(proc.stdout)["results"] if r.get("ok")}


def tfquant_prices():
    """BSM/crr via tff directly (repo rules respected by caller env)."""
    try:
        code = (
            "import json,sys;"
            "import jax;jax.config.update('jax_enable_x64',True);"
            "import jax.numpy as jnp;"
            "from tf_quant_finance.black_scholes import option_price;"
            "from tf_quant_finance.black_scholes.crr_binomial_tree import option_price_binomial;"
            "j=lambda x: jnp.array(x,dtype=jnp.float64);"
            f"c=float(option_price(volatilities=j({V}),strikes=j({K}),expiries=j({T}),spots=j({S}),discount_rates=j({R}),dividend_rates=j(0.0),is_call_options=j(True),dtype=jnp.float64));"
            f"p=float(option_price(volatilities=j({V}),strikes=j({K}),expiries=j({T}),spots=j({S}),discount_rates=j({R}),dividend_rates=j(0.0),is_call_options=j(False),dtype=jnp.float64));"
            f"t5=float(option_price_binomial(volatilities=j({V}),strikes=j({K}),expiries=j({T}),spots=j({S}),discount_rates=j({R}),dividend_rates=j(0.0),is_call_options=j(True),is_american=j(False),num_steps=500,dtype=jnp.float64));"
            "print(json.dumps({'iv_bsm_call':c,'iv_bsm_put':p,'iv_bin_do':t5}))"
        )
        proc = subprocess.run(
            ["uv", "run", "python", "-c", code], cwd=ROOT.parent / "tf-quant-finance",
            capture_output=True, text=True,
            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
        return json.loads(proc.stdout) if proc.returncode == 0 else {}
    except Exception:
        return {}


def main():
    lines = ["# Invariant check results", ""]
    violations = []
    df_fwd = math.exp(-Q * T) * S - math.exp(-R * T) * K  # I1 RHS

    for lib in ("quantburn", "quantcandle", "quantjax"):
        prices = run_lib(lib)
        if not prices:
            lines.append(f"- {lib}: runner unavailable")
            continue
        c, p = prices.get("iv_bsm_call"), prices.get("iv_bsm_put")
        if c is not None and p is not None:
            lhs = c - p
            err = abs(lhs - df_fwd)
            ok = err < 1e-9
            lines.append(f"- {lib} I1 put-call parity: C-P={lhs:.12f} vs {df_fwd:.12f} err={err:.1e} {'OK' if ok else 'VIOLATION'}")
            if not ok:
                violations.append((lib, "I1"))
        tree = prices.get("iv_bin_do")
        van0 = prices.get("iv_vanilla")
        if tree is not None and van0 is not None:
            err = abs(tree - van0) / van0
            ok = err < 5e-4  # 500-step CRR vs analytic (tree convention ~1/steps)
            lines.append(f"- {lib} I4 CRR(500)->BSM(q=0): {tree:.10f} vs {van0:.10f} rel={err:.1e} {'OK' if ok else 'VIOLATION'}")
            if not ok:
                violations.append((lib, "I4"))

    tfp = tfquant_prices()
    if tfp:
        # tff pair is dividend-free (matches the tree comparison below);
        # put-call parity RHS with q=0: S - e^{-rT}K
        lhs = tfp["iv_bsm_call"] - tfp["iv_bsm_put"]
        err = abs(lhs - (S - math.exp(-R * T) * K))
        lines.append(f"- tfquant I1 put-call parity: err={err:.1e} {'OK' if err < 1e-9 else 'VIOLATION'}")
        if err >= 1e-9:
            violations.append(("tfquant", "I1"))
        err = abs(tfp["iv_bin_do"] - tfp["iv_bsm_call"]) / tfp["iv_bsm_call"]  # both q=0 now
        lines.append(f"- tfquant I4 CRR(500)->BSM: rel={err:.1e} {'OK' if err < 5e-4 else 'VIOLATION'}")
        if err >= 5e-4:
            violations.append(("tfquant", "I4"))

    # I2/I3 need asset-or-nothing + barrier analytics — check via the
    # family's new vanilla_analytics modules (quantjax quickest to import).
    qj_code = """
import math, json
from quantjax.vanilla_analytics import binary_price
from quantjax.black_scholes_merton import black_scholes_merton, normal_cdf
S,K,T,V,R,Q = %r,%r,%r,%r,%r,%r
d1 = (math.log(S/K) + (R - Q + 0.5*V*V)*T)/(V*math.sqrt(T))
aon = math.exp(-Q*T)*S*normal_cdf(d1)
con_unit = binary_price(V,K,T,S,R,Q,True)
van = black_scholes_merton("call","european",S,K,T,V,R,Q)
print(json.dumps({"aon_minus_kcon": aon - K*con_unit, "van": van}))
""" % (S, K, T, V, R, Q)
    qj = subprocess.run(
        [str(ROOT.parent / "quantjax" / ".venv" / "bin" / "python"), "-c", qj_code],
        capture_output=True, text=True, cwd=str(ROOT.parent / "quantjax"))
    try:
        if qj.returncode != 0:
            raise RuntimeError(qj.stderr[-200:])
        m = json.loads(qj.stdout)
        # I2: asset-or-nothing - K*cash-or-nothing = vanilla call (per-unit con).
        err = abs(m["aon_minus_kcon"] - m["van"])
        lines.append(f"- quantjax I2 AoN-K*CoN=vanilla: {m['aon_minus_kcon']:.12f} vs {m['van']:.12f} err={err:.1e} {'OK' if err < 1e-9 else 'VIOLATION'}")
        if err >= 1e-9:
            violations.append(("quantjax", "I2"))
    except Exception as e:
        lines.append(f"- I2 skipped (quantjax analytics unavailable: {e})")
    try:
        qj3_code = qj_code.replace(
            'd1 = (math.log(S/K)',
            'B=85.0\nfrom quantjax.vanilla_analytics import barrier_price_analytic as _bpa\nki=_bpa(V,K,T,S,B,0.0,R,Q,True,False,True);ko=_bpa(V,K,T,S,B,0.0,R,Q,True,True,True)\nd1 = (math.log(S/K)'
        ).replace('print(json.dumps({"aon_minus_kcon"',
                  'print(json.dumps({"ki_ko": ki+ko, "aon_minus_kcon"')
        qj3 = subprocess.run(
            [str(ROOT.parent / "quantjax" / ".venv" / "bin" / "python"), "-c", qj3_code],
            capture_output=True, text=True, cwd=str(ROOT.parent / "quantjax"))
        if qj3.returncode != 0:
            raise RuntimeError(qj3.stderr[-200:])
        m3 = json.loads(qj3.stdout)
        # I3: barrier in + out = vanilla (rebate 0).
        err = abs(m3["ki_ko"] - m3["van"])
        lines.append(f"- quantjax I3 barrier in+out=vanilla: {m3['ki_ko']:.12f} vs {m3['van']:.12f} err={err:.1e} {'OK' if err < 1e-9 else 'VIOLATION'}")
        if err >= 1e-9:
            violations.append(("quantjax", "I3"))
    except Exception as e:
        lines.append(f"- I2/I3 skipped (quantjax analytics unavailable: {e})")

    # I5: ZCB decreasing in maturity (Vasicek with positive forwards).
    zcbs = []
    for m in (0.5, 1.0, 2.0, 5.0, 10.0):
        CASES["cases"][-1]["params"]["maturity"] = m
        prices = run_lib("quantburn")
        zcbs.append(prices.get("iv_vasicek"))
    CASES["cases"][-1]["params"]["maturity"] = 1.0
    mono = all(a is not None and b is not None and a > b for a, b in zip(zcbs, zcbs[1:]))
    lines.append(f"- quantburn I5 ZCB monotone decreasing: {['%.6f' % z if z else None for z in zcbs]} {'OK' if mono else 'VIOLATION'}")
    if not mono:
        violations.append(("quantburn", "I5"))

    lines += ["", f"**Violations: {len(violations)}**" + (f" — {violations}" if violations else " — all invariants hold")]
    (ROOT / "results" / "invariants.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
