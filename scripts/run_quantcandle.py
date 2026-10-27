#!/usr/bin/env python
"""Run the canonical cases through the quantcandle crosscheck bin.
Usage: scripts/run_quantcandle.py [--features mkl]  (writes results/quantcandle-<backend>.json)
"""
import json, subprocess, sys
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
    input=json.dumps(cases), capture_output=True, text=True, check=True)
out = json.loads(proc.stdout)
(ROOT / "results" / f"quantcandle-{backend}.json").write_text(proc.stdout)
ok = sum(1 for r in out["results"] if r["ok"])
print(f"quantcandle[{backend}]: {ok}/{len(out['results'])} ok -> results/quantcandle-{backend}.json")
