"""Small helpers shared across first-party world implementations."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def map_payload_to_json_str(payload: dict[str, Any]) -> str:
    """Serialize a map payload dict (may contain numpy arrays) to a JSON string."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return obj

    return json.dumps(_convert(payload))
