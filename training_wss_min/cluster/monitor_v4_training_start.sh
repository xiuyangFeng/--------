#!/bin/bash
# Read-only start monitor, scheduled ten minutes after the target GPU job starts.
set -euo pipefail

TARGET_JOB="${1:?target job id required}"
ERR_LOG="${2:?stderr training log required}"
EXPECTED_TRAIN="${3:?expected train count required}"

echo "checked=$(date) target_job=${TARGET_JOB} expected_train=${EXPECTED_TRAIN}"
test -s "$ERR_LOG"
grep -q "INFO device=cuda" "$ERR_LOG"
grep -q "INFO train cases=${EXPECTED_TRAIN} val cases=0" "$ERR_LOG"
grep -q "INFO epoch 0/400 loss=" "$ERR_LOG"

EPOCH_LINES="$(grep -c 'INFO epoch ' "$ERR_LOG")"
if [ "$EPOCH_LINES" -lt 2 ]; then
    echo "error: fewer than two epoch records after monitoring window" >&2
    exit 1
fi

if grep -Ein '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|oom|out of memory|traceback|runtimeerror|valueerror|filenotfound|wrong frame' "$ERR_LOG"; then
    echo "error: training failure signature found" >&2
    exit 1
fi

tail -n 5 "$ERR_LOG"
echo "monitor_status=passed epoch_records=${EPOCH_LINES}"
