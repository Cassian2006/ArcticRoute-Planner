from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import xarray as xr  # type: ignore
except Exception:
    xr = None  # type: ignore

from logging_config import get_logger

try:
    from scripts.build_cache_index import build_index
except ImportError:  # pragma: no cover
    from .build_cache_index import build_index  # type: ignore

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 删除优先级：临时/快照/中间产物 → 块级缓存 → 成品(merged/cost) → 输出(summary/route) → 报告
_TYPE_PRIORITY = {
    "snapshot": 0,
    "block": 1,
    "merged": 2,
    "cost": 2,
    "summary": 3,
    "route": 3,
    "report": 4,
}

_RENEWABLE = {"snapshot", "block", "merged", "cost", "summary", "route"}


def _now_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


@dataclass
class CleanParams:
    keep_months: int = 6
    max_size_gb: Optional[float] = None
    types: Optional[List[str]] = None
    protect_months: Optional[List[str]] = None
    base_dir: Optional[Path] = None
    index_path: Optional[Path] = None
    dry_run: bool = True
    yes: bool = False
    # handle guard
    stabilize_seconds: float = 4.0
    retry_open: int = 0
    try_open: bool = True
    # trash/safe delete
    use_trash: bool = True
    trash_ttl_days: Optional[int] = None
    purge_now: bool = False


def _load_entries(params: CleanParams) -> Dict:
    # 优先从索引文件读，否则现场扫描
    if params.index_path and Path(params.index_path).exists():
        idx_path = Path(params.index_path)
        obj = json.loads(idx_path.read_text(encoding="utf-8"))
        return obj
    # 现场扫描（不写文件）
    payload = build_index(
        write_json=False,
        base_dir=str(params.base_dir) if params.base_dir else None,
        include_outputs=True,
        include_reports=True,
        do_hash=False,
    )
    return payload


def _parse_types(s: Optional[str]) -> Optional[set[str]]:
    if not s:
        return None
    return {t.strip() for t in s.split(",") if t.strip()}


def _size_sum(entries: Iterable[dict]) -> int:
    return sum(int(e.get("size") or e.get("size_bytes") or 0) for e in entries)


def _month_key(ym: Optional[str]) -> Tuple[int, int]:
    if not ym or not (len(ym) == 6 and ym.isdigit()):
        return (0, 0)
    return (int(ym[:4]), int(ym[4:]))


def _to_relative(p: str) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(Path(p))


def plan_clean(params: CleanParams) -> Dict:
    payload = _load_entries(params)
    entries: List[dict] = list(payload.get("entries", []))

    # 归一字段
    for e in entries:
        if "size" not in e and "size_bytes" in e:
            e["size"] = e["size_bytes"]
        if "path" in e:
            e["path"] = _to_relative(e["path"])
        e.setdefault("type", "block")
        e.setdefault("ym", None)

    protect = set(params.protect_months or [])
    type_filter = set(params.types) if params.types else None

    # 最近 K 个月与保护集
    months = sorted({e.get("ym") for e in entries if e.get("ym")}, key=_month_key)
    recent_keep: set[str] = set(months[-params.keep_months:]) if params.keep_months and months else set()

    # 初步候选：不在保护集且类型命中过滤器
    candidates: List[dict] = []
    kept: List[dict] = []
    for e in entries:
        ym = e.get("ym")
        typ = str(e.get("type") or "block").lower()
        if type_filter and typ not in type_filter:
            kept.append(e)
            continue
        if ym in protect or (ym and ym in recent_keep):
            kept.append(e)
            continue
        # 可删除候选
        e["priority"] = _TYPE_PRIORITY.get(typ, 99)
        e["renewable"] = 1 if typ in _RENEWABLE else 0
        candidates.append(e)

    # 根据优先级与时间/大小排序：优先更可再生、更早（旧）、更大
    candidates.sort(key=lambda x: (x.get("priority", 99), _month_key(x.get("ym")), -int(x.get("size", 0))))

    total_bytes = _size_sum(entries)
    current_bytes = total_bytes

    selected: List[dict] = []
    # 若有 max_size_gb，需要删到阈值以下；否则全选候选
    if params.max_size_gb and params.max_size_gb > 0:
        limit = int(params.max_size_gb * (1024 ** 3))
        for e in candidates:
            if current_bytes <= limit:
                break
            selected.append(e)
            current_bytes -= int(e.get("size", 0))
    else:
        selected = candidates

    reclaimed = _size_sum(selected)
    plan = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            "keep_months": params.keep_months,
            "max_size_gb": params.max_size_gb,
            "types": sorted(list(type_filter)) if type_filter else None,
            "protect": sorted(list(protect)) if protect else [],
            "base_dir": str(params.base_dir) if params.base_dir else None,
            "index_path": str(params.index_path) if params.index_path else None,
            "dry_run": params.dry_run,
        },
        "totals": {
            "total_files": len(entries),
            "total_bytes": total_bytes,
            "reclaimed_bytes": reclaimed,
            "remaining_bytes": total_bytes - reclaimed,
            "selected_count": len(selected),
        },
        "entries": [
            {
                "path": e.get("path"),
                "ym": e.get("ym"),
                "type": e.get("type"),
                "size": int(e.get("size", 0)),
                "priority": int(e.get("priority", 99)),
                "renewable": int(e.get("renewable", 0)),
            }
            for e in selected
        ],
    }
    return plan


def _safe_move_to_trash(src: Path, trash_root: Path) -> bool:
    try:
        rel = src.resolve()
    except Exception:
        rel = src
    # 将绝对路径映射到 trash 下的相对结构
    rel_str = None
    try:
        rel_str = str(rel.relative_to(PROJECT_ROOT))
    except Exception:
        rel_str = rel.as_posix().lstrip("/").replace(":/", "/").replace(":\\", "/")
    dst = trash_root / rel_str
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        logger.warning("move to trash failed: %s -> %s: %s", src, dst, e)
        return False


def _is_stable_for_delete(path: Path, stabilize_seconds: float, try_open: bool, retry_open: int) -> bool:
    if not path.exists():
        return False
    if try_open and xr is not None:
        for _ in range(max(0, int(retry_open)) + 1):
            try:
                with xr.open_dataset(path) as _ds:
                    return True
            except Exception:
                time.sleep(0.5)
    # 尺寸稳定性
    try:
        s0 = path.stat().st_size
    except Exception:
        return False
    t0 = time.time()
    while True:
        time.sleep(0.5)
        try:
            s1 = path.stat().st_size
        except Exception:
            return False
        if s1 != s0:
            s0 = s1
            t0 = time.time()
            continue
        if time.time() - t0 >= stabilize_seconds:
            return True


def _purge_trash_older_than(days: Optional[int]) -> List[str]:
    removed: List[str] = []
    if not days or days <= 0:
        return removed
    root = PROJECT_ROOT / ".trash"
    if not root.exists():
        return removed
    now = time.time()
    ttl = days * 86400
    for p in root.glob("*"):
        try:
            if p.is_dir():
                mt = p.stat().st_mtime
                if now - mt > ttl:
                    shutil.rmtree(p, ignore_errors=True)
                    removed.append(str(p))
        except Exception:
            pass
    return removed


def execute_plan(plan: Dict, *, yes: bool = False, soft_delete: bool = True, stabilize_seconds: float = 4.0, try_open: bool = True, retry_open: int = 0, use_trash: bool = True, trash_ttl_days: Optional[int] = None, purge_now: bool = False) -> Dict:
    entries = plan.get("entries", [])
    if not entries:
        return {"deleted": 0, "failed": 0, "bytes": 0, "skipped": 0, "purged": []}
    if not yes:
        logger.info("执行未确认（--yes 未给出），跳过删除。")
        return {"deleted": 0, "failed": 0, "bytes": 0, "skipped": 0, "purged": []}

    trash_dir = PROJECT_ROOT / ".trash" / plan.get("generated_at", _now_ts()).replace(":", "").replace("-", "").replace("T", "_")
    trash_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = trash_dir / "trash_manifest.json"
    manifest: List[Dict] = []

    deleted = 0
    failed = 0
    skipped = 0
    bytes_sum = 0
    for e in entries:
        p = PROJECT_ROOT / e["path"] if not Path(e["path"]).is_absolute() else Path(e["path"])  # type: ignore
        try:
            size = int(e.get("size", 0))
        except Exception:
            size = 0
        # Handle Guard
        try:
            if not _is_stable_for_delete(p, stabilize_seconds, try_open, int(retry_open)):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        if soft_delete or use_trash:
            ok = _safe_move_to_trash(p, trash_dir)
            if ok:
                manifest.append({"src": str(e.get("path")), "dst_root": str(trash_dir), "size": size})
        else:
            try:
                os.remove(p)
                ok = True
            except Exception as err:
                logger.warning("delete failed: %s: %s", p, err)
                ok = False
        if ok:
            deleted += 1
            bytes_sum += size
        else:
            failed += 1
    # 记录回收站 manifest
    try:
        manifest_path.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "entries": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    # TTL 清理或立即清空
    purged = []
    if purge_now:
        purged = _purge_trash_older_than(0)
    else:
        purged = _purge_trash_older_than(trash_ttl_days)
    return {"deleted": deleted, "failed": failed, "bytes": bytes_sum, "skipped": skipped, "purged": purged}


__all__ = [
    "CleanParams",
    "plan_clean",
    "execute_plan",
]

