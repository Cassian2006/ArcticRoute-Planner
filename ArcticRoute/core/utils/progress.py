from __future__ import annotations

from dataclasses import dataclass, asdict
from time import time
import json
import os
import threading
import queue
from typing import Optional, Dict, Any

# 可选依赖
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    import multiprocessing as mp
except Exception:
    mp = None  # type: ignore


@dataclass
class ProgressEvent:
    ts: float
    kind: str              # "run_start"|"block_start"|"pixel_tick"|"block_done"|"run_done"|"health"
    label: Optional[str] = None
    block_idx: Optional[int] = None
    blocks_total: Optional[int] = None
    pixels_done: Optional[int] = None
    pixels_total: Optional[int] = None
    ok: Optional[int] = None
    skip: Optional[int] = None
    mem: Optional[float] = None      # %
    cpu: Optional[float] = None      # %
    msg: Optional[str] = None


class ProgressReporter:
    def __init__(self, mode: str, save_dir: str, label: str, blocks_total: int, interval: float = 2.0, use_tqdm: bool = True, mp_queue: Any | None = None):
        # mode: "off"|"console"|"both" ； both==console+jsonl
        self.mode = mode
        self.save_dir = save_dir
        self.label = label
        self.path = os.path.join(save_dir, f"progress_{label}.jsonl")
        os.makedirs(save_dir, exist_ok=True)
        self.interval = float(interval)
        self.blocks_total = int(blocks_total)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.use_tqdm = bool(use_tqdm)
        self._tb = None  # tqdm bar
        # 事件队列（跨进程）：优先使用传入 mp_queue，否则创建 Manager().Queue
        if mp_queue is not None:
            self.mq = mp_queue
        else:
            try:
                self._manager = mp.Manager() if mp is not None else None
                self.mq = (self._manager.Queue(maxsize=10000) if self._manager is not None else queue.Queue(maxsize=10000))
            except Exception:
                self._manager = None
                self.mq = queue.Queue(maxsize=10000)
        # ETA 估计缓存
        self._done_ts: list[float] = []

    def start(self):
        # truncate on new run
        open(self.path, "w").close()
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._thread.join(timeout=3)
        except Exception:
            pass
        if self._tb:
            try:
                self._tb.close()
            except Exception:
                pass
        # 释放 manager
        if hasattr(self, "_manager") and getattr(self, "_manager") is not None:
            try:
                self._manager.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass

    def emit(self, **kwargs):
        # 主进程/任意进程都可调用：直接往 mq 放 dict，_loop 统一封装为 ProgressEvent
        payload = dict(kwargs)
        payload.setdefault("ts", time())
        try:
            self.mq.put_nowait(payload)
        except Exception:
            try:
                self.mq.put(payload, timeout=0.1)
            except Exception:
                pass

    # 供子进程直接调用的静态方法（避免导入实例）
    @staticmethod
    def emit_to_queue(mq, **kwargs):
        try:
            payload = dict(kwargs)
            payload.setdefault("ts", time())
            mq.put_nowait(payload)
        except Exception:
            try:
                mq.put(payload, timeout=0.1)
            except Exception:
                pass

    def _loop(self):
        last_flush = 0.0
        processed_blocks = 0
        # tqdm 初始化
        if self.mode in ("console", "both") and self.use_tqdm and _HAS_TQDM:
            try:
                self._tb = tqdm(total=self.blocks_total, desc="Blocks", dynamic_ncols=True)
            except Exception:
                self._tb = None
        else:
            self._tb = None

        while not self._stop.is_set():
            try:
                ev_raw = self.mq.get(timeout=0.2)
            except Exception:
                continue

            # 统一为 ProgressEvent
            if isinstance(ev_raw, dict):
                ev = ProgressEvent(**({"ts": ev_raw.get("ts", time())} | ev_raw))
            else:
                # 兼容直接传入 ProgressEvent
                try:
                    ev = ProgressEvent(**asdict(ev_raw))  # type: ignore[arg-type]
                except Exception:
                    continue

            # 写入 jsonl
            try:
                with open(self.path, "a", encoding="utf-8") as wf:
                    wf.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
            except Exception:
                pass

            # 统计完成时间用于 ETA
            if ev.kind == "block_done":
                self._done_ts.append(ev.ts)
                # 只保留最近10分钟
                cutoff = ev.ts - 600.0
                self._done_ts = [t for t in self._done_ts if t >= cutoff]

            # 控制台显示
            if self.mode in ("console", "both"):
                if self._tb and ev.kind == "block_done":
                    processed_blocks += 1
                    try:
                        self._tb.update(1)
                    except Exception:
                        pass
                    postfix: Dict[str, Any] = {}
                    if ev.ok is not None and ev.skip is not None:
                        postfix["ok"] = ev.ok; postfix["skip"] = ev.skip
                    if ev.cpu is not None and ev.mem is not None:
                        postfix["cpu"] = f"{ev.cpu:.0f}%"; postfix["mem"] = f"{ev.mem:.0f}%"
                    # 吞吐与 ETA
                    tp, eta_txt = self._throughput_eta(processed_blocks)
                    postfix["tp"] = f"{tp:.2f} blk/min" if tp > 0 else "--"
                    postfix["eta"] = eta_txt
                    try:
                        self._tb.set_postfix(postfix, refresh=True)
                    except Exception:
                        pass
                elif (not self._tb) and ev.kind in ("pixel_tick", "block_done", "health"):
                    line = f"[{ev.kind}]"
                    if ev.block_idx is not None and ev.blocks_total is not None:
                        line += f" block {ev.block_idx}/{ev.blocks_total}"
                    if ev.pixels_done is not None and ev.pixels_total is not None:
                        line += f" pixels {ev.pixels_done}/{ev.pixels_total}"
                    if ev.ok is not None and ev.skip is not None:
                        line += f" ok={ev.ok} skip={ev.skip}"
                    if ev.cpu is not None and ev.mem is not None:
                        line += f" cpu={ev.cpu:.0f}% mem={ev.mem:.0f}%"
                    # ETA 文本
                    tp, eta_txt = self._throughput_eta(processed_blocks)
                    line += f" tp={tp:.2f} blk/min eta={eta_txt}"
                    print(line, flush=True)

    def _throughput_eta(self, processed_blocks: int) -> tuple[float, str]:
        # 近窗吞吐（块/分钟）与 ETA 文本
        now = time()
        cutoff = now - 600.0
        self._done_ts = [t for t in self._done_ts if t >= cutoff]
        n = len(self._done_ts)
        if n <= 1:
            return 0.0, "--"
        dur = max(1e-6, self._done_ts[-1] - self._done_ts[0])
        tp = n / (dur / 60.0)
        remain = max(0, self.blocks_total - processed_blocks)
        if tp <= 1e-6:
            return 0.0, "--"
        eta_min = remain / tp
        # 简易格式化
        if eta_min < 1.0:
            return tp, f"{int(eta_min*60)}s"
        elif eta_min < 60:
            return tp, f"{eta_min:.1f}m"
        else:
            hours = int(eta_min // 60)
            mins = int(eta_min % 60)
            return tp, f"{hours}h{mins:02d}m"

