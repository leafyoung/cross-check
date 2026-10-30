#!/usr/bin/env python
"""Run the canonical cases through the quantcandle crosscheck bin.
Usage: scripts/run_quantcandle.py [--features mkl]  (writes results/quantcandle-<backend>.json)
"""
import json
import sys, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
features = []
if "--features" in sys.argv:
    features = ["--features", sys.argv[sys.argv.index("--features") + 1]]
    backend = "mkl"
else:
    backend = "cpu"

cases = json.loads((ROOT / "cases" / "cases.json").read_text())
proc = subprocess.run(
    ["cargo", "run", "-q", *features, "--manifest-path", str(ROOT.parent / "quantcandle" / "Cargo.toml"),
     "--bin", "crosscheck", "--", "--backend", backend],
    input=json.dumps(cases), capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stdout[-3000:], file=sys.stderr)
    print(proc.stderr[-3000:], file=sys.stderr)
    raise SystemExit(f"{name}: runner failed rc={proc.returncode} (stderr above)")
out = json.loads(proc.stdout)
(ROOT / "results" / f"quantcandle-{backend}.json").write_text(proc.stdout)
ok = sum(1 for r in out["results"] if r["ok"])
print(f"quantcandle[{backend}]: {ok}/{len(out['results'])} ok -> results/quantcandle-{backend}.json")
