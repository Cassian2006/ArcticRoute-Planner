from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from sklearn.neighbors import NearestNeighbors  # type: ignore

from ArcticRoute.core.prior.transformer_split import make_splits

ROOT = Path(__file__).resolve().parents[2]
AOUT = ROOT / "ArcticRoute" / "data_processed" / "ais"
PROUT = ROOT / "ArcticRoute" / "data_processed" / "prior"
REPORTS = ROOT / "reports" / "phaseE"


@dataclass
class SelectConfig:
    ym: str
    c_min: float = 0.7
    d_max_nm: float = 5.0
    tau: float = 0.5


def _read_parquet_any(p: Path):
    if pl is not None:
        return pl.read_parquet(str(p))  # type: ignore
    return pd.read_parquet(str(p))  # type: ignore


def _to_pd(df_any: Any) -> "pd.DataFrame":  # type: ignore
    if pd is None:
        raise RuntimeError("pandas required")
    if pl is not None and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        return df_any.to_pandas()  # type: ignore
    if isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        return df_any
    raise RuntimeError("unsupported df type")


def _load_metrics(ym: str) -> Dict[str, Any]:
    p = REPORTS / f"prior_metrics_{ym}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _coverage_from_nc(ym: str, nc_path: Path, tau: float, seed: int = 42, max_points: int = 200_000) -> float:
    if xr is None:
        raise ImportError("xarray required to read prior raster")
    ds = xr.open_dataset(nc_path)
    lat = ds.get("lat").values
    lon = ds.get("lon").values
    P = ds.get("P_prior").values
    H, W = P.shape
    # flatten
    pts = np.column_stack([lat.reshape(-1), lon.reshape(-1)])
    Pflat = P.reshape(-1)
    nbrs = NearestNeighbors(n_neighbors=1).fit(pts)
    # sample val points from tracks (same month)
    seg = _to_pd(_read_parquet_any(AOUT / f"segment_index_{ym}.parquet"))
    trk = _to_pd(_read_parquet_any(AOUT / f"tracks_{ym}.parquet")).sort_values(["segment_id", "ts"])  # type: ignore
    splits = make_splits(seg, seed=seed, train_ratio=0.8, stratify="mmsi")
    val_mmsi = set(splits.get("val_mmsi", []))
    val_trk = trk[trk["mmsi"].isin(val_mmsi)][["lat", "lon"]].copy()
    if len(val_trk) > max_points:
        val_trk = val_trk.sample(n=max_points, random_state=seed)
    Q = val_trk.to_numpy(dtype=float)
    _, idx = nbrs.kneighbors(Q, return_distance=True)
    pvals = Pflat[idx.reshape(-1)]
    cov = float((pvals >= float(tau)).mean()) if len(pvals) else 0.0
    return cov


def select_prior(cfg: SelectConfig) -> Tuple[Dict[str, Any], Path, Path | None]:
    """根据评测与备选 prior，选择 adopt=transformer|density|none，并生成报告与拷贝选中层。"""
    ym = cfg.ym
    metrics = _load_metrics(ym)
    cov = float(metrics.get("coverage", -1.0))
    dev_mean_m = float(metrics.get("deviation_mean_m", 1e9))
    adopt = "none"
    reason = ""

    # 阈值判定（nm->m）
    if cov >= float(cfg.c_min) and dev_mean_m <= float(cfg.d_max_nm) * 1852.0:
        adopt = "transformer"
        reason = f"coverage {cov:.3f} >= {cfg.c_min}, deviation_mean {dev_mean_m/1852.0:.2f}nm <= {cfg.d_max_nm}nm"
    else:
        # 尝试密度骨架作为保底（若存在）
        dens_nc = PROUT / f"prior_density_{ym}.nc"
        if dens_nc.exists():
            cov_d = _coverage_from_nc(ym, dens_nc, tau=cfg.tau)
            if cov_d >= float(cfg.c_min) * 0.9:
                adopt = "density"
                reason = f"transformer not pass; density coverage {cov_d:.3f} >= {cfg.c_min*0.9:.3f}"
            else:
                adopt = "none"
                reason = f"both failed: transformer(cov={cov:.3f},dev_mean_m={dev_mean_m:.1f}), density(cov={cov_d:.3f})"
        else:
            adopt = "none"
            reason = f"transformer not pass and prior_density missing"

    # 输出：选中层拷贝至 prior_corridor_selected_YYYYMM.nc（附 attrs['source']）
    selected_nc: Path | None = None
    src_path = None
    if adopt == "transformer":
        src_path = PROUT / f"prior_transformer_{ym}.nc"
    elif adopt == "density":
        src_path = PROUT / f"prior_density_{ym}.nc"
    if src_path is not None and src_path.exists() and xr is not None:
        ds = xr.open_dataset(src_path)
        # 附增加 attrs
        ds.attrs.update({
            "adopt": adopt,
            "source": src_path.name,
            "ym": ym,
            "c_min": float(cfg.c_min),
            "d_max_nm": float(cfg.d_max_nm),
            "tau": float(cfg.tau),
        })
        out_nc = PROUT / f"prior_corridor_selected_{ym}.nc"
        ds.to_netcdf(str(out_nc))
        selected_nc = out_nc
    
    # 报告
    REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS / f"PRIOR_SELECT_{ym}.md"
    lines = [
        f"# PRIOR 采纳判定（{ym}）",
        "",
        f"- adopt: {adopt}",
        f"- reason: {reason}",
        f"- thresholds: C_min={cfg.c_min}, D_max={cfg.d_max_nm}nm, tau={cfg.tau}",
        f"- transformer_metrics: coverage={cov:.3f}, deviation_mean_m={dev_mean_m:.1f}",
    ]
    if adopt != "transformer":
        dens_nc = PROUT / f"prior_density_{ym}.nc"
        if dens_nc.exists():
            try:
                cov_d = _coverage_from_nc(ym, dens_nc, tau=cfg.tau)
                lines.append(f"- density_coverage(tau={cfg.tau}): {cov_d:.3f}")
            except Exception:
                pass
        else:
            lines.append("- density: missing")
    if adopt == "none":
        lines.append("\n## 建议\n- 提升训练/采样质量；\n- 调整簇参数或带宽分位；\n- 或准备 prior_density 作为保底。")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    result = {
        "ym": ym,
        "adopt": adopt,
        "reason": reason,
        "selected": (str(selected_nc) if selected_nc else None),
        "report": str(md_path),
    }
    return result, md_path, (selected_nc or None)


__all__ = ["SelectConfig", "select_prior"]












