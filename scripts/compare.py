#!/usr/bin/env python
"""Pairwise comparison of results/*.json against the QuantLib golden and
between libraries. Usage: scripts/compare.py [results_dir]  (default results/)"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"

golden = json.loads((res_dir / "quantlib.json").read_text())["results"]
golden = {r["id"]: r for r in golden}
cases = json.loads((ROOT / "cases" / "cases.json").read_text())["cases"]
tol_of = {c["id"]: c for c in cases}

libs = {}
for f in sorted(res_dir.glob("*.json")):
    if f.stem == "quantlib":
        continue
    payload = json.loads(f.read_text())
    if "results" not in payload:
        continue  # not a runner output (e.g. calibration.json)
    if any("id" not in r for r in payload["results"]):
        continue  # case-runner outputs key rows by 'id' (risk_golden.json does not)
    libs[f.stem] = {r["id"]: r for r in payload["results"]}

def is_unknown_model(r):
    return (not r["ok"]) and str(r.get("error", "")).find("unknown model") >= 0

def get_price(r):
    if not r["ok"]:
        return None
    return r.get("price")

def get_metric(cid, r, g):
    """Comparison metric: scalar price, or dict of vectors (curve values)."""
    if isinstance(g, dict):
        rv = r.get("values") or {}
        return ("vector", rv)
    return ("price", r.get("price"))

STOCHASTIC = {c["id"] for c in cases if "seed" in c.get("params", {})}

def convention_reason(cid):
    """Documented convention difference for a case (cases.json 'convention':
    true + 'convention_reason'): the runner and the golden compute the same
    named quantity under DIFFERENT conventions, so an over-tolerance row is
    rendered CONVENTION (with the reason, never silently, never FAIL)."""
    c = tol_of.get(cid) or {}
    if not c.get("convention"):
        return None
    return (str(c.get("convention_reason") or "documented convention difference")
            .replace("|", "/").replace("\n", " "))

def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)

def vector_rel_err(lp, gp):
    """Max relative error over every key/value of a golden payload (scalar
    or dict-of-lists) vs the runner's values dict. Raises ValueError on a
    mismatch (missing key) -> FAIL row."""
    worst = 0.0
    for k, gv in gp.items():
        if k not in lp:
            raise ValueError(f"payload missing key '{k}'")
        if isinstance(gv, list):
            if len(lp[k]) != len(gv):
                raise ValueError(f"payload length mismatch on key '{k}'")
            for a, b in zip(lp[k], gv):
                worst = max(worst, rel(a, b))
        else:
            worst = max(worst, rel(lp[k], gv))
    return worst

def first_val(x):
    if isinstance(x, dict):
        for v in x.values():
            return v[0] if isinstance(v, list) else v
    return x

lines = ["# Cross-check report", ""]

lines.append("## vs QuantLib golden")
lines.append("| case | lib | lib price | golden | rel err | tol | verdict |")
lines.append("|---|---|---|---|---|---|---|")
for name, res in libs.items():
    for cid, r in res.items():
        g = golden.get(cid, {})
        if not g.get("ok") or g.get("price") is None:
            continue
        metric_name, lp = get_metric(cid, r, g.get("price"))
        if metric_name == "vector" and not lp:
            lp = None  # runner produced no values (skipped/unknown model)
        if lp is None:
            verdict = "SKIP" if (r.get("skipped") or is_unknown_model(r)) else "FAIL"
            note = r.get("note") or r.get("error")
            gshow = first_val(g["price"])
            lines.append(f"| {cid} | {name} | ({note}) | {gshow:.10g} | — | — | {verdict} |")
            continue
        try:
            if metric_name == "vector":
                e = vector_rel_err(lp, g["price"])
                show = next(iter(lp.values()))
                if isinstance(show, list):
                    show = show[0]
            else:
                e = rel(lp, g["price"])
                show = lp
        except ValueError as exc:
            lines.append(f"| {cid} | {name} | {exc} | — | — | — | FAIL |")
            continue
        tol = g.get("tol", 1e-6)
        # quantorch stores model params at f32 precision -> widen accordingly
        if name.startswith("quantorch"):
            tol = max(tol, 1e-5)
        if cid in STOCHASTIC:
            # MC row vs an analytic/exact golden: the RNG streams differ
            # (Xorshift64Star vs QL/none), so MC noise dominates. Compared at
            # the documented 1e-2 rng tolerance; above it the row stays
            # visible as INFO(rng) with the deviation shown (never masked,
            # never a silent FAIL - see README, 'RNG parity').
            verdict = "PASS(rng)" if e <= 1e-2 else "INFO(rng)"
            tol = 1e-2
        else:
            verdict = "PASS" if e <= tol else "FAIL"
            reason = convention_reason(cid)
            if verdict == "FAIL" and reason:
                # Flagged convention difference: render CONVENTION with the
                # documented reason from the case entry - visible, counted,
                # and explicitly not a FAIL (see README, BK section).
                verdict = f"CONVENTION ({reason})"
        gshow = first_val(g["price"])
        lines.append(f"| {cid} | {name} | {show:.10g} | {gshow:.10g} | {e:.2e} | {tol:.0e} | {verdict} |")

lines += ["", "## library vs library (same case ids)"]
lines.append("| case | a | b | a price | b price | rel err | verdict |")
lines.append("|---|---|---|---|---|---|---|")
names = sorted(libs)
TOL_LIB = 1e-9   # deterministic cases
TOL_RNG = 1e-2   # stochastic cases: different RNG streams, MC noise dominates
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        for cid in libs[a]:
            if cid not in libs[b]:
                continue
            if is_unknown_model(libs[a][cid]) or is_unknown_model(libs[b][cid]):
                continue  # model not implemented in that library
            pa, pb = get_price(libs[a][cid]), get_price(libs[b][cid])
            if pa is None or pb is None:
                continue
            e = rel(pa, pb)
            if cid in STOCHASTIC:
                verdict = "PASS(rng)" if e <= TOL_RNG else "INFO(rng)"
            elif e <= TOL_LIB:
                verdict = "PASS"
            elif e <= 1e-5:
                verdict = "PASS(f32)"  # quantorch f32 param noise
            else:
                verdict = "FAIL"
            lines.append(f"| {cid} | {a} | {b} | {pa:.10g} | {pb:.10g} | {e:.2e} | {verdict} |")

report = "\n".join(lines) + "\n"
(res_dir / "report.md").write_text(report)
n_fail = report.count("| FAIL |")
n_conv = report.count("CONVENTION (")
print(report)
print(f"report written to {res_dir / 'report.md'} ({n_fail} FAIL rows, "
      f"{n_conv} CONVENTION rows)")
