#!/usr/bin/env bash
# Full cross-check pipeline: golden + all runners + compare. Per-step failure
# capture (a failing runner doesn't abort the rest); prints the final tally.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
TFQ=../tf-quant-finance
declare -a FAILED=()

step() { # step <name> <cmd...>
  local name=$1; shift
  echo "=== $name"
  if ! "$@"; then FAILED+=("$name"); echo "!!! $name FAILED (continuing)"; fi
}

step golden        $PY scripts/golden_quantlib.py
step quantburn-cpu $PY scripts/run_quantburn.py
step quantburn-vulkan timeout 1500 $PY scripts/run_quantburn.py --features vulkan
step quantcandle-cpu $PY scripts/run_quantcandle.py
step quantcandle-mkl  $PY scripts/run_quantcandle.py --features mkl
step quantorch     ../quantorch/.venv/bin/python scripts/run_quantorch.py

echo "=== tfquant (from tf-quant repo, repo rules)"
if (cd $TFQ && XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$TFQ nice -n 19 \
      uv run python ../cross-check/scripts/run_tfquant.py); then :; else FAILED+=(tfquant); echo "!!! tfquant FAILED (continuing)"; fi

echo "=== quantjax-cpu"
QJ=$PWD/../quantjax
CC=$PWD/cases/cases.json
CPOUT=$PWD/results/quantjax-cpu.json
if (cd $QJ && XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m quantjax.cli \
      --cases $CC --backend cpu > $CPOUT); then :;
else FAILED+=(quantjax-cpu); echo "!!! quantjax-cpu FAILED (continuing)"; fi

echo "=== quantjax-rocm (tf-quant uv env, low memory)"
ROCM_OUT=$PWD/results/quantjax-rocm.json
cat > /tmp/qj_rocm_all.py <<'PYEOF'
import sys
sys.path.insert(0, "/var/home/yangye/devv/qfin/pricing/quantjax")
sys.argv = ["cli", "--backend", "rocm",
            "--cases", "/var/home/yangye/devv/qfin/pricing/cross-check/cases/cases.json"]
from quantjax.cli import main
main()
PYEOF
if (cd $TFQ && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
      nice -n 19 uv run --project . python /tmp/qj_rocm_all.py \
      > "$ROCM_OUT"); then :; else FAILED+=(quantjax-rocm); echo "!!! quantjax-rocm FAILED (continuing)"; fi

step invariants $PY scripts/invariants.py > /dev/null
# calibration_check is phase-based; run all phases in sequence
step calibration-generate timeout 300 $PY scripts/calibration_check.py generate
step calibration-quantorch timeout 900 $PY scripts/calibration_check.py quantorch
step calibration-qlfit timeout 300 $PY scripts/calibration_check.py qlfit
step calibration-report timeout 120 $PY scripts/calibration_check.py report
step compare $PY scripts/compare.py > /dev/null
$PY scripts/compare.py 2>/dev/null | tail -1 || true

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL STEPS OK"
else
  echo "FAILED STEPS: ${FAILED[*]}"
fi
grep -cE '\| PASS|\| PASS\(rng\)|\| PASS\(f32\)' results/report.md | xargs echo "PASS rows:"
grep -cE '\| FAIL \|' results/report.md | xargs echo "FAIL rows:"
grep -cE '\| SKIP \|' results/report.md | xargs echo "SKIP rows:"
