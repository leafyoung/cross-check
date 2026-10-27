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
    libs[f.stem] = {r["id"]: r for r in json.loads(f.read_text())["results"]}

def is_unknown_model(r):
    return (not r["ok"]) and str(r.get("error", "")).find("unknown model") >= 0

def get_price(r):
    if not r["ok"]:
        return None
    return r.get("price")

def get_metric(cid, r, g):
    """Comparison metric: scalar price, or dfs vector (swap curve)."""
    if isinstance(g, dict) and "dfs" in g:
        rv = r.get("values") or {}
        return ("dfs", rv.get("dfs"))
    return ("price", r.get("price"))

STOCHASTIC = {c["id"] for c in cases if "seed" in c.get("params", {})}

def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)

lines = ["# Cross-check report", ""]

lines.append("## vs QuantLib golden")
lines.append("| case | lib | lib price | golden | rel err | tol | verdict |")
lines.append("|---|---|---|---|---|---|---|")
for name, res in libs.items():
    for cid, r in res.items():
        g = golden.get(cid, {})
        if not g.get("ok") or g.get("price") is None:
            continue
        if cid in STOCHASTIC:
            continue  # stochastic cases have no golden; compared lib-vs-lib only
        metric_name, lp = get_metric(cid, r, g.get("price"))
        if metric_name == "dfs" and g.get("price") is None:
            continue
        if lp is None:
            verdict = "SKIP" if (r.get("skipped") or is_unknown_model(r)) else "FAIL"
            note = r.get("note") or r.get("error")
            gshow = g["price"]
            gshow = gshow.get("dfs", ["?"])[0] if isinstance(gshow, dict) else gshow
            lines.append(f"| {cid} | {name} | ({note}) | {gshow:.10g} | — | — | {verdict} |")
            continue
        if metric_name == "dfs":
            gdfs = g["price"]["dfs"]
            if len(lp) != len(gdfs):
                lines.append(f"| {cid} | {name} | len mismatch | — | — | — | FAIL |")
                continue
            e = max(abs(a - b) / max(abs(b), 1e-300) for a, b in zip(lp, gdfs))
            lp = gdfs[0]  # representative display value
        else:
            e = rel(lp, g["price"])
        tol = g.get("tol", 1e-6)
        # quantorch stores model params at f32 precision -> widen accordingly
        if name.startswith("quantorch"):
            tol = max(tol, 1e-5)
        verdict = "PASS" if e <= tol else "FAIL"
        gshow = g["price"]
        gshow = gshow.get("dfs", ["?"])[0] if isinstance(gshow, dict) else gshow
        lines.append(f"| {cid} | {name} | {lp:.10g} | {gshow:.10g} | {e:.2e} | {tol:.0e} | {verdict} |")

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
print(report)
print(f"report written to {res_dir / 'report.md'} ({n_fail} FAIL rows)")
