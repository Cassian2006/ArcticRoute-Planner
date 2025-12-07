# REUSE: 基于 apps.app_min 中的最小任务管理实现进行桥接与增强
from __future__ import annotations
import atexit
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# 线程池（限制并发 2~4）
_MAX_WORKERS = int(os.environ.get("AR_UI_TASK_WORKERS", "3"))
_EXECUTOR = ThreadPoolExecutor(max_workers=max(2, min(4, _MAX_WORKERS)))
_LOCK = threading.Lock()

# 持久化目录与文件

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ui_out_dir() -> Path:
    d = _repo_root()/"outputs"/"ui"
    d.mkdir(parents=True, exist_ok=True)
    (d/"logs").mkdir(parents=True, exist_ok=True)
    (d/"actions").mkdir(parents=True, exist_ok=True)
    return d


def _tasks_jsonl_path() -> Path:
    return _ui_out_dir()/"tasks.jsonl"


@dataclass
class Task:
    id: str
    name: str
    kind: str
    fn: Optional[Callable[..., Any]] = None
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued/running/succeeded/failed/canceled
    started: Optional[float] = None
    ended: Optional[float] = None
    progress: float = 0.0
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    _future: Optional[Future] = None
    _cancel_flag: bool = False

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "started": self.started,
            "ended": self.ended,
            "progress": self.progress,
            "stdout": self.stdout_path,
            "stderr": self.stderr_path,
            "result_path": self.result_path,
            "meta": self.meta,
            "error": self.error,
        }


_TASKS: Dict[str, Task] = {}


# 简易 stdout/stderr 捕获到文件
class _Tee:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, s: str) -> None:
        if not s:
            return
        with self._lock:
            self._fh.write(s)
            self._fh.flush()

    def flush(self) -> None:
        try:
            with self._lock:
                self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            with self._lock:
                self._fh.close()
        except Exception:
            pass


def _persist_append(task: Task) -> None:
    try:
        line = json.dumps(task.to_public(), ensure_ascii=False)
        _tasks_jsonl_path().write_text("", encoding="utf-8") if not _tasks_jsonl_path().exists() else None
        with open(_tasks_jsonl_path(), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _persist_load(limit: int = 200) -> List[Dict[str, Any]]:
    p = _tasks_jsonl_path()
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
        out: List[Dict[str, Any]] = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _update_snapshot(task: Task) -> None:
    # 附加一条当前状态快照（便于崩溃恢复）
    _persist_append(task)


def _runtime_yaml() -> Path:
    return _repo_root()/"ArcticRoute"/"config"/"runtime.yaml"


def _read_runtime_flag(default_bg: bool = True, default_persist: bool = True) -> Tuple[bool, bool]:
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(_runtime_yaml().read_text(encoding="utf-8")) or {}
        ui = (data.get("ui") or {})
        return bool(ui.get("task_background", default_bg)), bool(ui.get("task_persist", default_persist))
    except Exception:
        return default_bg, default_persist


def submit_task(fn: Callable[..., Any], *, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] | None = None, name: str = "task", kind: str = "generic", meta: Dict[str, Any] | None = None) -> str:
    """
    提交后台任务。
    参数：fn(recorded_fn) -> 可调用；args/kwargs 传递给 fn；meta.inputs 建议包含页面参数。
    返回 task_id。
    """
    if kwargs is None:
        kwargs = {}
    if meta is None:
        meta = {}
    task_id = uuid.uuid4().hex[:12]
    ui_dir = _ui_out_dir()
    log_path = ui_dir/"logs"/f"{task_id}.log"
    err_path = log_path  # stdout/stderr 同一文件，便于快速定位

    rec = Task(id=task_id, name=name, kind=kind, fn=fn, args=args, kwargs=kwargs, meta=meta,
               stdout_path=str(log_path), stderr_path=str(err_path))
    with _LOCK:
        _TASKS[task_id] = rec

    def _runner():
        bg_enabled, persist_enabled = _read_runtime_flag()
        rec.status = "running"
        rec.started = time.time()
        _update_snapshot(rec)
        tee_out = _Tee(log_path)
        tee_err = _Tee(err_path)
        try:
            # 将 stdout/stderr 临时重定向
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = tee_out, tee_err  # type: ignore
            try:
                result_path = fn(*args, **kwargs, task_ctx={"task_id": task_id, "progress": _set_progress(rec)})  # type: ignore
                rec.result_path = str(result_path) if result_path else rec.result_path
                rec.status = "succeeded" if not rec._cancel_flag else "canceled"
            finally:
                sys.stdout, sys.stderr = old_out, old_err
        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
        finally:
            rec.ended = time.time()
            _update_snapshot(rec)
            tee_out.close(); tee_err.close()

    rec._future = _EXECUTOR.submit(_runner)
    return task_id


def _set_progress(task: Task) -> Callable[[float], None]:
    def _inner(p: float) -> None:
        with _LOCK:
            task.progress = max(0.0, min(100.0, float(p)))
        _update_snapshot(task)
    return _inner


def get_task(task_id: str) -> Dict[str, Any] | None:
    with _LOCK:
        t = _TASKS.get(task_id)
    if t is not None:
        return t.to_public()
    # 若为新进程（崩溃恢复），从 jsonl 尝试返回最后一次快照
    snaps = [s for s in _persist_load(500) if s.get("id") == task_id]
    return snaps[-1] if snaps else None


def list_tasks(kind: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with _LOCK:
        for t in _TASKS.values():
            items.append(t.to_public())
    # 合并历史（去重，优先内存态）
    hist = _persist_load(limit * 5)
    seen = {i["id"] for i in items}
    for s in reversed(hist):  # 从旧到新推进，确保最终为最新快照
        if kind and s.get("kind") != kind:
            continue
        if s.get("id") in seen:
            continue
        items.append(s)
        seen.add(s.get("id"))
    # 过滤 kind
    if kind:
        items = [x for x in items if x.get("kind") == kind]
    # 排序 & 截断
    items.sort(key=lambda x: x.get("started") or 0, reverse=True)
    return items[:limit]


def cancel_task(task_id: str) -> bool:
    with _LOCK:
        t = _TASKS.get(task_id)
        if not t:
            return False
        t._cancel_flag = True
        # 软取消：无法安全中断时，fn 应检查 task_ctx.progress 或 _cancel_flag
        fut = t._future
    try:
        if fut and not fut.done():
            # 尝试不强制取消，标记为 canceled，具体中断由 fn 自行检查
            return True
        return True
    finally:
        _update_snapshot(t) if t else None


def prune_tasks(max_age_hours: int = 24) -> int:
    cutoff = time.time() - max(1, max_age_hours) * 3600
    removed = 0
    with _LOCK:
        for k in list(_TASKS.keys()):
            t = _TASKS[k]
            if t.ended and t.ended < cutoff:
                del _TASKS[k]
                removed += 1
    # 不清理 jsonl，以保留历史
    return removed


# 退出时清理线程池
@atexit.register
def _shutdown_executor():
    try:
        _EXECUTOR.shutdown(wait=False, cancel_futures=False)
    except Exception:
        pass

