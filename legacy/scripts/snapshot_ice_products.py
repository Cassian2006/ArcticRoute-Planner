#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-04-06 · 文件竞争与快照策略（Watcher/Copy）

目标：避免读取“正在写”的文件。检测训练产物（整月 nc 或分块目录）在尺寸稳定 ≥ N 秒或尝试打开成功后，
采用临时名 .part 复制，并用 os.replace 原子落盘到 merged/ 下的快照目录，供后续 merge/route 等步骤读取。

用法示例：
  python scripts/snapshot_ice_products.py --ym 202412
  python scripts/snapshot_ice_products.py --ym 202412 --stabilize-seconds 4 --timeout 300

输出：
  单行 JSON 打印到 stdout，包含 ym/mode/copied/dest_dir/files 等字段；失败时打印到 stderr 并非零退出。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# xarray 仅在 try_open 时需要
try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    xr = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DEFAULT = REPO_ROOT / "ArcticRoute" / "data_processed" / "ice_forecast"


def _find_inputs(base: Path, ym: str) -> Tuple[str, List[Path]]:
    # 优先 blocks
    for sub in ("blocks", "_blocks"):
        bdir = base / sub / ym
        if bdir.is_dir():
            files = sorted(p for p in bdir.glob("block_*.nc") if p.is_file())
            if files:
                return ("blocks", files)
    # 其次整月
    mfile = base / f"ice_forecast_{ym}.nc"
    if mfile.exists():
        return ("monthly", [mfile])
    return ("none", [])


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1


def _is_stable_file(path: Path, stabilize_seconds: float, try_open: bool) -> bool:
    if not path.exists():
        return False
    # 优先尝试打开
    if try_open and xr is not None:
        try:
            with xr.open_dataset(path) as _ds:
                pass
            return True
        except Exception:
            pass
    # 检查 N 秒尺寸稳定
    s0 = _size(path)
    if s0 <= 0:
        return False
    t0 = time.time()
    while True:
        time.sleep(0.5)
        s1 = _size(path)
        if s1 != s0:
            # 尺寸改变，重置计时
            s0 = s1
            t0 = time.time()
            continue
        if time.time() - t0 >= stabilize_seconds:
            return True


def _wait_stable(path: Path, stabilize_seconds: float, timeout: float, try_open: bool) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_stable_file(path, stabilize_seconds, try_open):
            return True
        time.sleep(0.5)
    return False


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    # 确保旧的 .part 不残留
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    # 复制到临时文件
    with src.open("rb") as r, tmp.open("wb") as w:
        shutil.copyfileobj(r, w, length=16 * 1024 * 1024)
        w.flush()
        os.fsync(w.fileno())
    # 原子替换
    os.replace(tmp, dst)


def snapshot_month(ym: str, base: Path, stabilize_seconds: float, timeout: float, try_open: bool = True) -> Dict:
    mode, inputs = _find_inputs(base, ym)
    if mode == "none":
        raise FileNotFoundError(
            f"未找到输入：{base}/(blocks|_blocks)/{ym}/block_*.nc 或 {base}/ice_forecast_{ym}.nc"
        )

    out_dir = base / "merged"
    files_out: List[str] = []

    if mode == "monthly":
        src = inputs[0]
        if not _wait_stable(src, stabilize_seconds, timeout, try_open):
            raise TimeoutError(f"文件未稳定（超时）：{src}")
        dst = out_dir / f"_snapshot_ice_forecast_{ym}.nc"
        _atomic_copy(src, dst)
        files_out.append(str(dst))
    else:
        # blocks 模式：逐个文件快照
        snap_dir = out_dir / "_blocks_snapshot" / ym
        copied = 0
        for src in inputs:
            if not _wait_stable(src, stabilize_seconds, timeout, try_open):
                raise TimeoutError(f"文件未稳定（超时）：{src}")
            dst = snap_dir / src.name
            _atomic_copy(src, dst)
            files_out.append(str(dst))
            copied += 1

    return {
        "ym": ym,
        "mode": mode,
        "copied": len(files_out),
        "dest_dir": str(out_dir if mode == "monthly" else (out_dir / "_blocks_snapshot" / ym)),
        "files": files_out,
        "stabilize_seconds": stabilize_seconds,
        "timeout": timeout,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="P1-04-06 文件竞争与快照策略（Watcher/Copy）")
    ap.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    ap.add_argument(
        "--base-dir",
        default=str(BASE_DEFAULT),
        help="基础目录（包含 blocks 与 ice_forecast_* 的目录）",
    )
    ap.add_argument("--stabilize-seconds", type=float, default=4.0, help="尺寸稳定秒数阈值")
    ap.add_argument("--timeout", type=float, default=300.0, help="最大等待秒数")
    ap.add_argument("--no-try-open", action="store_true", help="不使用尝试打开作为稳定性判定")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    ym = args.ym
    base = Path(args.base_dir)
    try_open = not bool(args.no_try_open)
    try:
        result = snapshot_month(ym, base, float(args.stabilize_seconds), float(args.timeout), try_open=try_open)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    except TimeoutError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    except PermissionError as e:
        print(f"[ERROR] 权限错误：{e}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] 未预期错误：{e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

















