from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore
import torch

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.core.prior.transformer_model import TrajectoryTransformer, ModelConfig
from ArcticRoute.core.prior.transformer_feats import FeatureConfig, make_features, apply_posenc

from ArcticRoute.core.prior.transformer_train import load_ckpt  # reuse loader util

from ArcticRoute.cache.index_util import register_artifact


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "ais"


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


def _load_model(ckpt_path: Path, d_in: int = 9) -> TrajectoryTransformer:
    cfg = ModelConfig()
    model = TrajectoryTransformer(d_in=d_in, cfg=cfg)
    # build dummy optimizer/scheduler/projector to satisfy load_ckpt API
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start_epoch, best = load_ckpt(ckpt_path, model=model, optimizer=optim, scheduler=None, projector=None, ema=None)
    model.eval()
    return model


def export_embeddings(ym: str, ckpt: str, seq_len: int = 512, batch_size: int = 32, device: str = "cuda") -> Path:
    """导出某月所有段的嵌入到 embeddings_<YYYYMM>.parquet。

    输出列：segment_id, mmsi, vclass, ts_start, ts_end, emb_0..emb_255
    """
    seg_path = OUT_DIR / f"segment_index_{ym}.parquet"
    if not seg_path.exists():
        # fallback to generic
        seg_path = OUT_DIR / "segment_index.parquet"
    trk_path = OUT_DIR / f"tracks_{ym}.parquet"
    if not seg_path.exists() or not trk_path.exists():
        raise FileNotFoundError(f"缺少 segment_index 或 tracks：{seg_path} / {trk_path}")

    df_seg = _to_pandas(_read_parquet_any(seg_path)).copy()
    df_trk = _to_pandas(_read_parquet_any(trk_path)).copy()
    df_trk = df_trk.sort_values(["segment_id", "ts"]).copy()

    # 分组索引
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for seg_id, g in df_trk.groupby("segment_id"):
        groups[str(seg_id)] = {
            "ts": g["ts"].to_numpy(dtype=np.int64, copy=True),
            "lat": g["lat"].to_numpy(dtype=np.float64, copy=True),
            "lon": g["lon"].to_numpy(dtype=np.float64, copy=True),
            "sog": g.get("sog", pd.Series([np.nan]*len(g))).to_numpy(dtype=np.float64, copy=True),
            "cog": g.get("cog", pd.Series([np.nan]*len(g))).to_numpy(dtype=np.float64, copy=True),
        }

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = _load_model(Path(ckpt), d_in=9).to(device_t)

    # 推理
    feat_cfg = FeatureConfig()
    rows: List[Dict[str, Any]] = []

    seg_rows = []
    for _, r in df_seg.iterrows():
        seg_rows.append((str(r["segment_id"]), int(r["mmsi"]), r.get("vclass", None), int(r["ts_start"]), int(r["ts_end"])) )

    # 批次化
    for i in range(0, len(seg_rows), batch_size):
        batch_meta = seg_rows[i:i+batch_size]
        xs: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        metas: List[Tuple[str,int,Any,int,int]] = []
        for (seg_id, mmsi, vclass, ts_s, ts_e) in batch_meta:
            tr = groups.get(seg_id)
            if not tr:
                continue
            feats = make_features(tr["lat"], tr["lon"], tr.get("sog"), tr.get("cog"), tr["ts"], feat_cfg)
            X = feats["x"]  # [t,C]
            M = feats["mask"]  # [t]
            # pad/trunc
            tlen = min(X.shape[0], seq_len)
            x_pad = np.zeros((seq_len, X.shape[1]), dtype=np.float32)
            m_pad = np.zeros((seq_len,), dtype=bool)
            if tlen > 0:
                x_pad[:tlen, :] = X[:tlen, :]
                m_pad[:tlen] = M[:tlen]
            xs.append(torch.from_numpy(x_pad))
            masks.append(torch.from_numpy(m_pad))
            metas.append((seg_id, mmsi, vclass, ts_s, ts_e))
        if not xs:
            continue
        x_t = torch.stack(xs, dim=0).to(device_t)
        m_t = torch.stack(masks, dim=0).to(device_t)
        posenc = apply_posenc(seq_len, method="rope")
        with torch.no_grad():
            z = model(x_t, mask=m_t, posenc=posenc).mean(dim=1)  # [B,256]
            z = z.detach().cpu().numpy().astype(np.float32)
        for j in range(z.shape[0]):
            seg_id, mmsi, vclass, ts_s, ts_e = metas[j]
            row = {
                "segment_id": seg_id,
                "mmsi": int(mmsi),
                "vclass": (None if vclass is None else str(vclass)),
                "ts_start": int(ts_s),
                "ts_end": int(ts_e),
            }
            emb = z[j]
            for k in range(emb.shape[0]):
                row[f"emb_{k}"] = float(emb[k])
            rows.append(row)

    # 写 parquet
    out_path = OUT_DIR / f"embeddings_{ym}.parquet"
    if pd is None:
        raise RuntimeError("pandas required to write embeddings parquet")
    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(str(out_path), engine="pyarrow")

    # 统计
    E = df_out[[c for c in df_out.columns if c.startswith("emb_")]].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(E, axis=1)
    mean_norm = float(np.mean(norms)) if len(norms) else 0.0
    mean_val = float(np.mean(E)) if E.size > 0 else 0.0
    stats = {
        "segments": int(len(df_seg)),
        "exported": int(len(df_out)),
        "mean_norm": mean_norm,
        "mean_val": mean_val,
    }
    print(json.dumps({"ym": ym, "out": str(out_path), **stats}, ensure_ascii=False))

    try:
        register_artifact(run_id=Path(ckpt).stem, kind="prior_embeddings", path=str(out_path), attrs={"ym": ym, **stats})
    except Exception:
        pass

    # DOD：行数=段数（若有缺失轨迹则可能略少；打印提示）
    if len(df_out) != len(df_seg):
        print(f"[WARN] 段计数与嵌入导出不一致: segments={len(df_seg)} exported={len(df_out)}")

    return out_path


__all__ = ["export_embeddings"]


