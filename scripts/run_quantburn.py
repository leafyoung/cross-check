#!/usr/bin/env python
"""Run the canonical cases through the quantburn crosscheck bin.
Usage: scripts/run_quantburn.py [--features vulkan]  (writes results/quantburn-<backend>.json)
"""
import json
import sys, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
features = []
backend = "cpu"
if "--features" in sys.argv:
    f = sys.argv[sys.argv.index("--features") + 1]
    features = ["--features", f]
    backend = f

cases = json.loads((ROOT / "cases" / "cases.json").read_text())
proc = subprocess.run(
    ["cargo", "run", "-q", *features, "--manifest-path", str((ROOT.parent / "quantburn" / "Cargo.toml")
                        if (ROOT.parent / "quantburn" / "Cargo.toml").exists()
                        else (ROOT / "quantburn" / "Cargo.toml")),
     "--bin", "crosscheck", "--", "--backend", backend],
    input=json.dumps(cases), capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stdout[-3000:], file=sys.stderr)
    print(proc.stderr[-3000:], file=sys.stderr)
    raise SystemExit(f"{name}: runner failed rc={proc.returncode} (stderr above)")
out = json.loads(proc.stdout)
(ROOT / "results" / f"quantburn-{backend}.json").write_text(proc.stdout)
ok = sum(1 for r in out["results"] if r["ok"])
print(f"quantburn[{backend}]: {ok}/{len(out['results'])} ok -> results/quantburn-{backend}.json")
