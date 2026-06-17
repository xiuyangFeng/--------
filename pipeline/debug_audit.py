from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

_DEBUG_LOG_PATH = Path("/public/newhome/cy/Digital_twin/GNN/.cursor/debug-1a8747.log")
_SESSION_ID = "1a8747"
_COUNTS: Dict[str, int] = {}


def agent_debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Dict[str, Any],
    max_count: int = 5,
) -> None:
    key = f"{run_id}:{hypothesis_id}:{location}:{message}"
    count = _COUNTS.get(key, 0)
    if count >= max_count:
        return
    _COUNTS[key] = count + 1
    payload = {
        "sessionId": _SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
