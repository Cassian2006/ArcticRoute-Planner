from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import math
import os
from pathlib import Path
import json

try:
    # 适配器：不改 Fuel-Model 源码的情况下做集成
    from .adapter import build_adapter_from_env  # type: ignore
except Exception:  # pragma: no cover
    build_adapter_from_env = None  # type: ignore

app = FastAPI(title="fuel-service", version="0.1.3")

# CORS：允许本地开发端口（5173/Vite、3000/React、8501/Streamlit）
_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局适配器（懒加载）
_ADAPTER = None


# --- 解析仓库根目录（minimum 根） ---
_THIS_FILE = Path(__file__).resolve()
# 向上查找 pyproject.toml 以定位仓根
_REPO_ROOT = None
for p in _THIS_FILE.parents:
    if (p / "pyproject.toml").exists() and (p / "README.md").exists():
        _REPO_ROOT = p
        break
if _REPO_ROOT is None:
    # 回退：三层向上
    _REPO_ROOT = _THIS_FILE.parents[3]


class LonLat(BaseModel):
    lon: float
    lat: float


class FuelPredictRequest(BaseModel):
    # 简化：接受 GeoJSON LineString 或 FeatureCollection（首个 LineString）
    route_geojson: Dict[str, Any] = Field(..., description="GeoJSON 对象：LineString/Feature(LineString)/FeatureCollection")
    ym: Optional[str] = Field(None, description="月份 YYYYMM，可选")
    vessel_class: Optional[str] = Field("cargo_iceclass", description="船型，默认 cargo_iceclass")


class SegmentResult(BaseModel):
    index: int
    length_nm: float
    fuel_tons: float
    co2_tons: float


class FuelPredictResponse(BaseModel):
    ok: bool
    total_length_nm: float
    total_fuel_tons: float
    total_co2_tons: float
    per_segment: List[SegmentResult]
    backend: str = "mock"


def _extract_linestring_coords(gj: Dict[str, Any]) -> List[LonLat]:
    t = gj.get("type")
    if t == "FeatureCollection":
        feats = gj.get("features") or []
        for ft in feats:
            geom = (ft or {}).get("geometry") or {}
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates") or []
                return [LonLat(lon=float(x), lat=float(y)) for x, y in coords]
    if t == "Feature":
        geom = gj.get("geometry") or {}
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates") or []
            return [LonLat(lon=float(x), lat=float(y)) for x, y in coords]
    if t == "LineString":
        coords = gj.get("coordinates") or []
        return [LonLat(lon=float(x), lat=float(y)) for x, y in coords]
    raise ValueError("route_geojson 必须为 LineString/Feature/FeatureCollection(LineString)")


def _haversine_nm(a: LonLat, b: LonLat) -> float:
    R_m = 6_371_000.0
    phi1 = math.radians(a.lat)
    phi2 = math.radians(b.lat)
    dphi = math.radians(b.lat - a.lat)
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    d_m = 2*R_m*math.atan2(math.sqrt(h), math.sqrt(1-h))
    return d_m / 1852.0


def _get_adapter():
    global _ADAPTER
    if _ADAPTER is None and build_adapter_from_env is not None:
        try:
            _ADAPTER = build_adapter_from_env()
        except Exception:
            _ADAPTER = None
    return _ADAPTER


@app.post("/fuel/predict", response_model=FuelPredictResponse)
async def predict(req: FuelPredictRequest):
    # 提取坐标
    coords = _extract_linestring_coords(req.route_geojson)
    if len(coords) < 2:
        return FuelPredictResponse(ok=False, total_length_nm=0.0, total_fuel_tons=0.0, total_co2_tons=0.0, per_segment=[], backend="mock")

    # 排放因子（与 ArcticRoute/config/eco.yaml 对齐默认 3.114）
    ef_co2 = 3.114  # t CO2 / t fuel

    # 计算分段距离
    seg_nm: List[float] = []
    total_nm = 0.0
    for i in range(len(coords) - 1):
        d_nm = _haversine_nm(coords[i], coords[i+1])
        seg_nm.append(d_nm)
        total_nm += d_nm

    # 优先尝试适配器（真实或规则版），按“每海里燃油吨数”估计
    backend = "mock"
    per_nm_from_adapter: Optional[float] = None
    adp = _get_adapter()
    if adp is not None:
        try:
            per_nm_from_adapter = adp.predict_per_nm([(c.lon, c.lat) for c in coords], req.ym, req.vessel_class or "cargo_iceclass")  # type: ignore[attr-defined]
        except Exception:
            per_nm_from_adapter = None
        if isinstance(per_nm_from_adapter, (float, int)) and per_nm_from_adapter > 0:
            backend = "fuel_adapter"

    # 若适配器不可用，则走内置基准（原 mock）
    if backend == "mock":
        base_fuel_per_nm = {
            "icebreaker": 0.020,
            "cargo_iceclass": 0.015,
            "cargo_standard": 0.012,
            "fishing_small": 0.006,
        }.get((req.vessel_class or "cargo_iceclass"), 0.015)
        per_nm = float(base_fuel_per_nm)
    else:
        per_nm = float(per_nm_from_adapter)  # type: ignore[arg-type]

    # 聚合输出
    per_seg: List[SegmentResult] = []
    total_fuel = 0.0
    for i, d_nm in enumerate(seg_nm):
        fuel = per_nm * d_nm
        co2 = fuel * ef_co2
        total_fuel += fuel
        per_seg.append(SegmentResult(index=i, length_nm=d_nm, fuel_tons=fuel, co2_tons=co2))

    return FuelPredictResponse(
        ok=True,
        total_length_nm=total_nm,
        total_fuel_tons=total_fuel,
        total_co2_tons=total_fuel * ef_co2,
        per_segment=per_seg,
        backend=backend,
    )


@app.get("/fuel/health")
async def health():
    adp = _get_adapter()
    info = {
        "ok": True,
        "service": "fuel-service",
        "version": "0.1.3",
        "adapter": {
            "enabled": bool(adp is not None),
            "model_path": os.environ.get("FUEL_MODEL_PATH"),
            "mode": os.environ.get("FUEL_MODEL_MODE", "sklearn"),
        },
        "repo_root": str(_REPO_ROOT),
        "cors": {"origins": _allowed_origins},
    }
    return info


# ------------------ 只读 REST：路由与图层元信息 ------------------

class RouteItem(BaseModel):
    name: str
    path: str
    size: int
    mtime: float


@app.get("/routes/list", response_model=List[RouteItem])
async def list_routes(ym: Optional[str] = None):
    routes_dir = _REPO_ROOT / "ArcticRoute" / "data_processed" / "routes"
    if not routes_dir.exists():
        return []
    items: List[RouteItem] = []
    for p in sorted(routes_dir.glob("*.geojson")):
        if ym and (f"_{ym}_" not in p.name):
            continue
        try:
            stat = p.stat()
            items.append(RouteItem(name=p.name, path=str(p), size=stat.st_size, mtime=stat.st_mtime))
        except Exception:
            continue
    return items


@app.get("/routes/get")
async def get_route(name: str):
    routes_dir = _REPO_ROOT / "ArcticRoute" / "data_processed" / "routes"
    p = (routes_dir / name)
    if not p.exists() or p.suffix.lower() != ".geojson":
        raise HTTPException(status_code=404, detail="route not found")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read route geojson")
    return data


class LayerMeta(BaseModel):
    name: str
    path: str
    exists: bool
    size: Optional[int] = None


@app.get("/layers/meta", response_model=List[LayerMeta])
async def layers_meta(ym: str):
    risk_dir = _REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
    pri_dir = _REPO_ROOT / "ArcticRoute" / "data_processed" / "prior"
    
    candidates = [
        ("risk_fused", risk_dir / f"risk_fused_{ym}.nc"),
        ("R_ice_eff", risk_dir / f"R_ice_eff_{ym}.nc"),
        ("prior_penalty", risk_dir / f"prior_penalty_{ym}.nc"),
        ("prior_centerlines", pri_dir / "centerlines" / f"prior_centerlines_{ym}.geojson"),
    ]
    out: List[LayerMeta] = []
    for name, path in candidates:
        if path.exists():
            try:
                size = path.stat().st_size
            except Exception:
                size = None
            out.append(LayerMeta(name=name, path=str(path), exists=True, size=size))
        else:
            out.append(LayerMeta(name=name, path=str(path), exists=False))
    return out


class ReportItem(BaseModel):
    name: str
    path: str
    exists: bool


@app.get("/reports/list", response_model=List[ReportItem])
async def reports_list(ym: str):
    base = _REPO_ROOT / "ArcticRoute" / "reports" / "d_stage"
    cands = [
        ("pareto_html", base / "phaseG" / f"pareto_{ym}_nsr_wbound_smoke.html"),
        ("calibration_json", base / "phaseH" / f"calibration_{ym}.json"),
        ("calibration_png", base / "phaseH" / f"calibration_{ym}.png"),
        ("robust_json", base / "phaseI" / f"uncertainty_{ym}.json"),
    ]
    items: List[ReportItem] = []
    for name, path in cands:
        items.append(ReportItem(name=name, path=str(path), exists=path.exists()))
    return items
