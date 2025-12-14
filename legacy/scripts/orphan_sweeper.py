from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from logging_config import get_logger

try:
    from scripts.build_cache_index import build_index
except ImportError:  # pragma: no cover
    from .build_cache_index import build_index  # type: ignore

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OrphanParams:
    grace_sec: float = 600.0
    base_dir: Optional[Path] = None
    index_path: Optional[Path] = None
    dry_run: bool = True
    yes: bool = False
    use_trash: bool = True


def _now_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _to_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def _load_entries(params: OrphanParams) -> Dict:
    if params.index_path and Path(params.index_path).exists():
        return json.loads(Path(params.index_path).read_text(encoding="utf-8"))
    return build_index(
        write_json=False,
        base_dir=str(params.base_dir) if params.base_dir else None,
        include_outputs=True,
        include_reports=True,
        do_hash=False,
    )


def _is_part_to_clean(path: Path, grace_sec: float) -> bool:
    if not path.name.endswith('.part'):
        return False
    try:
        mt = path.stat().st_mtime
    except Exception:
        return False
    return (time.time() - mt) > grace_sec


def _is_json_broken(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding='utf-8'))
        return False
    except Exception:
        return True


def _is_zip_broken(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, 'r') as z:
            bad = z.testzip()
            return bad is not None
    except Exception:
        return True


def _is_nc_broken(path: Path) -> bool:
    try:
        import xarray as xr  # type: ignore
    except Exception:
        # 无 xarray 时不判为损坏
        return False
    try:
        with xr.open_dataset(path) as _ds:
            return False
    except Exception:
        return True


def _group_key(e: dict) -> Optional[Tuple[str, str]]:
    ym = e.get('ym')
    typ = str(e.get('type') or 'block')
    if not ym:
        return None
    return ym, typ


def plan_orphans(params: OrphanParams) -> Dict:
    payload = _load_entries(params)
    entries: List[dict] = list(payload.get('entries', []))
    # 归一
    for e in entries:
        if 'size' not in e and 'size_bytes' in e:
            e['size'] = e['size_bytes']

    parts: List[dict] = []
    broken: List[dict] = []
    redundant: List[dict] = []

    # 1) .part 过期
    for e in entries:
        p = PROJECT_ROOT / e['path'] if not Path(e['path']).is_absolute() else Path(e['path'])
        if _is_part_to_clean(p, params.grace_sec):
            parts.append({"path": _to_rel(p), "reason": ".part_expired", "size": int(e.get('size', 0))})

    # 2) 重复（同 ym 同类型）
    groups: Dict[Tuple[str,str], List[dict]] = {}
    for e in entries:
        k = _group_key(e)
        if not k:
            continue
        groups.setdefault(k, []).append(e)
    for k, items in groups.items():
        if len(items) <= 1:
            continue
        # 选保留：mtime 最新（ties: size 最大）
        def _mtime(e):
            return e.get('mtime') or e.get('modified') or ''
        items_sorted = sorted(items, key=lambda x: (_mtime(x), int(x.get('size', 0))))
        keep = items_sorted[-1]
        for it in items_sorted[:-1]:
            redundant.append({
                "path": it['path'],
                "ym": it.get('ym'),
                "type": it.get('type'),
                "reason": "duplicate",
                "keep": keep['path'],
                "size": int(it.get('size', 0)),
            })

    # 3) 损坏文件
    for e in entries:
        p = PROJECT_ROOT / e['path'] if not Path(e['path']).is_absolute() else Path(e['path'])
        if not p.exists() or not p.is_file():
            continue
        name = p.name.lower()
        try:
            if name.endswith('.json') or name.endswith('.geojson'):
                if _is_json_broken(p):
                    broken.append({"path": _to_rel(p), "reason": "json_broken", "size": int(e.get('size', 0))})
            elif name.endswith('.zip'):
                if _is_zip_broken(p):
                    broken.append({"path": _to_rel(p), "reason": "zip_broken", "size": int(e.get('size', 0))})
            elif name.endswith('.nc'):
                if _is_nc_broken(p):
                    broken.append({"path": _to_rel(p), "reason": "nc_broken", "size": int(e.get('size', 0))})
        except Exception:
            # 防御性：异常即视为损坏
            broken.append({"path": _to_rel(p), "reason": "probe_failed", "size": int(e.get('size', 0))})

    # 合并候选
    selected = parts + redundant + broken
    reclaimed = sum(int(x.get('size', 0)) for x in selected)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {"grace_sec": params.grace_sec, "dry_run": params.dry_run},
        "counts": {"parts": len(parts), "redundant": len(redundant), "broken": len(broken)},
        "reclaimed_bytes": reclaimed,
        "entries": selected,
    }
    return report


def execute_orphan_plan(plan: Dict, *, yes: bool = False, use_trash: bool = True) -> Dict:
    entries = plan.get('entries', [])
    if not yes:
        logger.info("dry-run（未 --yes），跳过删除")
        return {"deleted": 0, "failed": 0, "bytes": 0}
    ts = plan.get('generated_at', _now_ts()).replace(':','').replace('-','').replace('T','_')
    trash_dir = PROJECT_ROOT / ".trash" / f"orph_{ts}"
    trash_dir.mkdir(parents=True, exist_ok=True)
    deleted = 0
    failed = 0
    bytes_sum = 0
    for e in entries:
        p = PROJECT_ROOT / e['path'] if not Path(e['path']).is_absolute() else Path(e['path'])
        size = int(e.get('size', 0))
        try:
            if use_trash:
                # 软删除：同 cache_cleaner 的策略
                from .cache_cleaner import _safe_move_to_trash  # type: ignore
                ok = _safe_move_to_trash(p, trash_dir)
            else:
                os.remove(p)
                ok = True
        except Exception as err:
            logger.warning("orphan delete failed: %s: %s", p, err)
            ok = False
        if ok:
            deleted += 1
            bytes_sum += size
        else:
            failed += 1
    return {"deleted": deleted, "failed": failed, "bytes": bytes_sum}


__all__ = ["OrphanParams", "plan_orphans", "execute_orphan_plan"]
















