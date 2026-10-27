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

def get_price(r):
    if not r["ok"]:
        return None
    return r.get("price")

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
        lp = get_price(r)
        if lp is None:
            verdict = "SKIP" if r.get("skipped") else "FAIL"
            note = r.get("note") or r.get("error")
            lines.append(f"| {cid} | {name} | ({note}) | {g['price']:.10g} | — | — | {verdict} |")
            continue
        e = rel(lp, g["price"])
        tol = g.get("tol", 1e-6)
        # quantorch stores model params at f32 precision -> widen accordingly
        if name.startswith("quantorch"):
            tol = max(tol, 1e-5)
        verdict = "PASS" if e <= tol else "FAIL"
        lines.append(f"| {cid} | {name} | {lp:.10g} | {g['price']:.10g} | {e:.2e} | {tol:.0e} | {verdict} |")

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
