from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from ArcticRoute.core.feedback.schema import load_jsonl, dedup, build_digest  # REUSE


def ingest_feedback(feedback_path: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    items = load_jsonl(feedback_path)
    items2 = dedup(items)
    digest = build_digest(items2)

    ts = os.path.splitext(os.path.basename(feedback_path))[0]
    out_json = os.path.join(out_dir, f"feedback_ingested_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"items": items2, "digest": digest}, f, ensure_ascii=False, indent=2)
    return {"items": items2, "digest": digest, "out": out_json}


__all__ = ["ingest_feedback"]
