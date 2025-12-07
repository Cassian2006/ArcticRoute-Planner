from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# 依赖：hdbscan, scikit-learn
try:
    import hdbscan  # type: ignore
except Exception as e:  # pragma: no cover
    hdbscan = None  # type: ignore

try:
    from sklearn.metrics import silhouette_score  # type: ignore
except Exception:  # pragma: no cover
    silhouette_score = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "ais"
REPORT_DIR = ROOT / "reports" / "phaseE"


def _read_parquet_any(p: Path):
    if pl is not None:
        return pl.read_parquet(str(p))  # type: ignore
    return pd.read_parquet(str(p))  # type: ignore


def _to_pandas(df_any: Any) -> "pd.DataFrame":  # type: ignore
    if pd is None:
        raise RuntimeError("pandas required")
    if pl is not None and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        return df_any.to_pandas()  # type: ignore
    if isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        return df_any
    raise RuntimeError("Unsupported DF type")


@dataclass
class ScanConfig:
    min_cluster_size: Sequence[int] = (30, 50, 80)
    min_samples: Sequence[int] = (5, 10, 15)
    metric: str = "euclidean"  # HDBSCAN 距离度量（emb 空间通常用欧式/余弦）


def _extract_embeddings(df: "pd.DataFrame") -> np.ndarray:  # type: ignore
    cols = [c for c in df.columns if str(c).startswith("emb_")]
    if not cols:
        raise ValueError("embeddings parquet 缺少 emb_* 列")
    X = df[cols].to_numpy(dtype=np.float32)
    return X


def _silhouette_safe(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    if silhouette_score is None:
        return float("nan")
    # 仅在至少 2 个簇且每簇>=2 时有效
    uniq = [int(x) for x in np.unique(labels) if x != -1]
    if len(uniq) < 2:
        return float("nan")
    # 仅对非噪声样本计算
    mask = labels != -1
    if mask.sum() < 3:
        return float("nan")
    try:
        return float(silhouette_score(X[mask], labels[mask], metric=metric))
    except Exception:
        return float("nan")


def scan_hdbscan(ym: str, cfg: ScanConfig = ScanConfig()) -> Dict[str, Any]:
    if hdbscan is None:
        raise ImportError("需要安装 hdbscan 与 scikit-learn：pip install hdbscan scikit-learn")

    emb_path = OUT_DIR / f"embeddings_{ym}.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"未找到嵌入：{emb_path}")

    df = _to_pandas(_read_parquet_any(emb_path))
    X = _extract_embeddings(df)
    N = X.shape[0]
    seg_ids = df["segment_id"].astype(str).tolist()

    results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    # 断点续跑：加载已完成进度
    progress_dir = REPORT_DIR / "cluster"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"scan_progress_{ym}.json"
    done_set: set[Tuple[int,int]] = set()
    if progress_path.exists():
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            for it in prog.get("done", []):
                try:
                    done_set.add((int(it[0]), int(it[1])))
                except Exception:
                    continue
            # 恢复累积的 results 与 best（meta，不带 labels）
            if isinstance(prog.get("results"), list):
                for it in prog["results"]:
                    if isinstance(it, dict):
                        results.append(it)
            if isinstance(prog.get("best"), dict):
                best = prog["best"]
        except Exception:
            pass

    # 进度心跳：每 30s 打印一次进度
    import time, threading, sys
    mcs_list = list(cfg.min_cluster_size)
    ms_list = list(cfg.min_samples)
    total = max(1, len(mcs_list) * len(ms_list))
    progress = {"done": 0, "total": total, "current": None, "start": time.time(), "stop": False}

    def _heartbeat():
        while not progress["stop"]:
            done = int(progress["done"])
            tot = int(progress["total"])
            cur = progress["current"]
            elapsed = int(time.time() - progress["start"])
            print(f"[PROG] prior.cluster {done}/{tot} ({(done/max(1,tot))*100:.1f}%) current={cur} elapsed={elapsed}s")
            sys.stdout.flush()
            time.sleep(30)

    th = threading.Thread(target=_heartbeat, daemon=True)
    th.start()

    try:
        for mcs in mcs_list:
            for ms in ms_list:
                pair = (int(mcs), int(ms))
                # 跳过已完成组合（断点续跑）
                if pair in done_set:
                    progress["done"] += 1
                    print(f"[SKIP] prior.cluster 已完成组合 mcs={mcs} ms={ms}")
                    continue
                progress["current"] = pair
                t0 = time.time()
                try:
                    clus = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms), metric=cfg.metric)
                    labels = clus.fit_predict(X)
                except Exception as e:
                    results.append({"min_cluster_size": mcs, "min_samples": ms, "error": str(e)})
                    progress["done"] += 1
                    # 持久化进度
                    try:
                        with open(progress_path, "w", encoding="utf-8") as f:
                            json.dump({"ym": ym, "done": list(done_set), "results": results, "best": (best if isinstance(best, dict) else None)}, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    continue
                noise_ratio = float((labels == -1).sum() / max(1, len(labels)))
                k = len([c for c in np.unique(labels) if c != -1])
                sil = _silhouette_safe(X, labels, metric=cfg.metric)
                # 简单综合评分：sil 优先；噪声惩罚；鼓励更多非噪声覆盖
                coverage = 1.0 - noise_ratio
                score = (0.7 * (0.0 if np.isnan(sil) else sil)) + (0.3 * coverage) - 0.2 * noise_ratio
                item = {
                    "min_cluster_size": int(mcs),
                    "min_samples": int(ms),
                    "clusters": int(k),
                    "noise_ratio": float(noise_ratio),
                    "silhouette": (None if np.isnan(sil) else float(sil)),
                    "score": float(score),
                    "time_sec": float(time.time() - t0),
                }
                # 统计簇规模
                sizes = []
                for cid in np.unique(labels):
                    if cid == -1:
                        continue
                    sizes.append(int((labels == cid).sum()))
                sizes.sort(reverse=True)
                item["sizes"] = sizes[:10]
                results.append(item)
                # 选择（满足噪声<0.4 优先）
                ok_noise = noise_ratio < 0.4
                cand = {**item, "labels": labels.tolist()}
                if best is None:
                    best = cand
                else:
                    best_ok = best.get("noise_ratio", 1.0) < 0.4
                    if ok_noise and not best_ok:
                        best = cand
                    elif ok_noise == best_ok and float(item["score"]) > float(best.get("score", -1e9)):
                        best = cand
                progress["done"] += 1
                done_set.add(pair)
                # 持久化进度（包含 best 的 meta 与 labels）
                try:
                    with open(progress_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "ym": ym,
                            "done": list(sorted(list(done_set))),
                            "results": results,
                            "best": ({k: v for k, v in (best or {}).items() if k != "labels"}),
                            "best_labels": (best.get("labels") if isinstance(best, dict) and "labels" in best else None),
                        }, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
    finally:
        progress["stop"] = True
        try:
            th.join(timeout=1.0)
        except Exception:
            pass

    # 写报告
    report_dir = REPORT_DIR / "cluster"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"scan_report_{ym}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"ym": ym, "results": results, "best": {k: v for k, v in (best or {}).items() if k != "labels"}}, f, ensure_ascii=False, indent=2)

    # 输出 cluster_assign.parquet
    assign_path = OUT_DIR / f"cluster_assign_{ym}.parquet"
    if best is None or "labels" not in best:
        # 无有效结果，输出全 -1
        lab = np.full(N, -1, dtype=int)
    else:
        lab = np.array(best["labels"], dtype=int)
    df_out = pd.DataFrame({"segment_id": seg_ids, "cluster_id": lab.astype(int)})
    df_out.to_parquet(str(assign_path), engine="pyarrow")

    try:
        register_artifact(run_id=ym, kind="prior_cluster_assign", path=str(assign_path), attrs={"ym": ym})
        register_artifact(run_id=ym, kind="prior_cluster_scan", path=str(report_path), attrs={"ym": ym})
    except Exception:
        pass

    # DOD：主簇数量、噪声占比
    noise_ratio = float((lab == -1).sum() / max(1, len(lab)))
    print(json.dumps({"ym": ym, "assign": str(assign_path), "report": str(report_path), "noise_ratio": noise_ratio}, ensure_ascii=False))
    return {"assign": str(assign_path), "report": str(report_path), "noise_ratio": noise_ratio}


__all__ = ["scan_hdbscan", "ScanConfig"]







