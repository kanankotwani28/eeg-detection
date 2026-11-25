#!/usr/bin/env bash
# Run the essential pipeline end to end on real data (phase 0 -> splits -> preprocessing -> essential experiments),
# exporting results and regenerating REPORT.md after every experiment so partial results are always available.
# Usage: TDBRAIN_PATH=... scripts/run_essential.sh [overlay]    (overlay defaults to "budget")
set -euo pipefail
OVERLAY="${1:-budget}"
PY="${PY:-uv run python}"
export MLFLOW_DISABLE_AGENT_HINT=1
$PY -m src.data.index
$PY -m src.data.splits
$PY -m src.preprocess.pipeline --cond restEC
for E in E01 E02 E16 E03 E04 E05 E06 E09 E10 E12 E11 E13; do
  echo "=== $E ($(date -u +%H:%M:%S)) ==="
  $PY -m src.experiments "$E" --overlay "$OVERLAY"
  $PY -m src.report.export
  $PY -m src.report.tables
done
echo "=== ALL ESSENTIAL EXPERIMENTS DONE ($(date -u +%H:%M:%S)) ==="
