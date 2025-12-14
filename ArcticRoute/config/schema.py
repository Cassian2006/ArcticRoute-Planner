from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, validator

ModelT = TypeVar("ModelT", bound=BaseModel)


def _model_validate(model_cls: Type[ModelT], data: Dict[str, Any]) -> ModelT:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # type: ignore[attr-defined]
    return model_cls.parse_obj(data)


def model_dump(model: BaseModel, *, exclude_none: bool = True) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=exclude_none)  # type: ignore[attr-defined]
    return model.dict(exclude_none=exclude_none)


def format_validation_error(err: ValidationError, source: str) -> str:
    parts: List[str] = []
    for issue in err.errors():
        location = ".".join(str(item) for item in issue.get("loc", ()))
        message = issue.get("msg", "")
        parts.append(f"- field `{location}`: {message}")
    details = "\\n".join(parts) if parts else str(err)
    return f"[SCHEMA] validation failed for {source}\\n{details}"


class DataSection(BaseModel):
    env_nc: str = Field(..., description="Risk environment NetCDF path")
    corridor_prob: Optional[str] = Field(None, description="Optional corridor probability path")
    accident_density_static: Optional[str] = Field(None, description="Optional static accident density path")
    stac_cache_dir: Optional[str] = Field(None, description="Directory for STAC cache")
    cog_dir: Optional[str] = Field(None, description="Directory for COG assets")
    sat_cache_dir: Optional[str] = Field(None, description="Directory for raster mosaics")


class RunSection(BaseModel):
    var: str = Field(..., description="Primary variable name in env dataset")
    tidx: int = Field(0, ge=0, description="Time index")
    time_step_nodes: int = Field(0, ge=0, description="Additional time-step layers")


class RouteSection(BaseModel):
    start: str = Field(..., description="Start coordinate lat,lon")
    goal: str = Field(..., description="Goal coordinate lat,lon")


class CostSection(BaseModel):
    beta: float = Field(..., description="Risk weight beta")
    p: float = Field(..., description="Risk exponent p")
    beta_a: Optional[float] = Field(None, description="Legacy accident weight alias")
    beta_accident: float = Field(0.0, description="Accident weight beta_accident")
    fuel_alpha: float = Field(0.3, ge=0.0, description="Fuel proxy weight")

    @validator("beta_accident", pre=True, always=True)
    def _merge_beta(cls, value: Optional[float], values: Dict[str, Any]) -> float:
        beta_a = values.get("beta_a")
        if beta_a is not None:
            return float(beta_a)
        if value is None:
            return 0.0
        return float(value)


class BehaviorSection(BaseModel):
    corridor_path: Optional[str] = Field(None, description="Corridor probability file")
    accident_path: Optional[str] = Field(None, description="Accident density file")
    accident_mode: Optional[str] = Field(None, description="Accident mode static/time")
    gamma: float = Field(0.0, description="Behavior weight gamma")

    @validator("accident_mode")
    def _validate_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        lowered = value.lower()
        if lowered not in {"static", "time"}:
            raise ValueError("accident_mode must be static or time")
        return lowered


class CropSection(BaseModel):
    bbox: Optional[List[float]] = Field(None, description="Crop bounds [N, W, S, E]")
    coarsen: Optional[int] = Field(None, ge=1, description="Coarsen factor")

    @validator("bbox")
    def _check_bbox(cls, value: Optional[Sequence[Any]]) -> Optional[List[float]]:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must contain 4 values (N, W, S, E)")
        return [float(item) for item in value]


class OutputsSection(BaseModel):
    dir: Optional[str] = Field(None, description="Output directory")
    geojson: Optional[str] = Field(None, description="GeoJSON output path")
    png: Optional[str] = Field(None, description="PNG output path")


class CVSatParams(BaseModel):
    mission: str = Field("S2", description="Satellite mission identifier")
    prefer_source: str = Field("CDSE", description="Preferred catalogue source")
    max_items: int = Field(6, ge=1, description="Maximum STAC items to request")
    cloud_mask: bool = Field(True, description="Whether to apply cloud mask")

    @validator("prefer_source")
    def _validate_source(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CDSE", "MPC"}:
            raise ValueError("prefer_source must be CDSE or MPC")
        return normalized


class DLIceParams(BaseModel):
    model_path: Optional[str] = Field("models/ice_unet.ts", description="Placeholder model path")
    batch_size: int = Field(4, ge=1, description="Batch size for inference")


class PredictorParams(BaseModel):
    cv_sat: Optional[CVSatParams] = None
    dl_ice: Optional[DLIceParams] = None


class RuntimeConfig(BaseModel):
    data: DataSection
    run: RunSection
    route: RouteSection
    cost: CostSection
    behavior: BehaviorSection
    crop: Optional[CropSection] = None
    output: Optional[OutputsSection] = None
    predictor: str = Field("env_nc", description="Primary predictor name")
    predictor_params: Optional[PredictorParams] = None
    alpha_ice: Optional[float] = Field(0.5, ge=0.0, description="Blend weight between risk and ice layers")

    @validator("predictor")
    def _validate_predictor(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"env_nc", "cv_sat", "dl_ice"}:
            raise ValueError("predictor must be env_nc, cv_sat, or dl_ice")
        return lowered

    class Config:
        extra = "forbid"


class ScenarioDefaults(BaseModel):
    env_nc: str
    var: Optional[str]
    beta: Optional[float]
    gamma: Optional[float]
    p: Optional[float]
    beta_a: Optional[float]
    beta_accident: Optional[float]
    fuel_alpha: Optional[float]
    tidx: Optional[int]
    time_step_nodes: Optional[int]
    bbox: Optional[List[float]]
    coarsen: Optional[int]
    start: Optional[str]
    goal: Optional[str]
    corridor: Optional[str]
    accident: Optional[str]
    accident_mode: Optional[str]
    output_dir: Optional[str]
    stac_cache_dir: Optional[str]
    cog_dir: Optional[str]
    sat_cache_dir: Optional[str]
    accident_density_static: Optional[str]
    predictor: Optional[str]
    predictor_params: Optional[Dict[str, Any]]
    alpha_ice: Optional[float]

    class Config:
        extra = "forbid"

    @validator("bbox")
    def _bbox_defaults(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("bbox must contain 4 values (N, W, S, E)")
        return [float(item) for item in value]


class ScenarioItem(BaseModel):
    name: str
    env_nc: Optional[str]
    var: Optional[str]
    beta: Optional[float]
    gamma: Optional[float]
    p: Optional[float]
    beta_a: Optional[float]
    beta_accident: Optional[float]
    fuel_alpha: Optional[float]
    tidx: Optional[int]
    time_step_nodes: Optional[int]
    bbox: Optional[List[float]]
    coarsen: Optional[int]
    start: Optional[str]
    goal: Optional[str]
    corridor: Optional[str]
    accident: Optional[str]
    accident_mode: Optional[str]
    output_dir: Optional[str]
    stac_cache_dir: Optional[str]
    cog_dir: Optional[str]
    sat_cache_dir: Optional[str]
    accident_density_static: Optional[str]
    predictor: Optional[str]
    predictor_params: Optional[Dict[str, Any]]
    alpha_ice: Optional[float]

    class Config:
        extra = "forbid"

    @validator("bbox")
    def _bbox_validate(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("bbox must contain 4 values (N, W, S, E)")
        return [float(item) for item in value]


class ScenariosConfig(BaseModel):
    defaults: Optional[ScenarioDefaults] = None
    scenarios: List[ScenarioItem]

    class Config:
        extra = "forbid"


def validate_runtime_config(data: Dict[str, Any]) -> RuntimeConfig:
    cfg = _model_validate(RuntimeConfig, data)

    # 额外校验：当选择 predictor=cv_sat 时，要求可选依赖已安装
    # 通过核心预测器聚合入口判断可用性（SatCVPredictor 为 None 代表导入失败/依赖缺失）
    if getattr(cfg, "predictor", "").lower() == "cv_sat":
        try:
            from ArcticRoute.core.predictors import SatCVPredictor  # type: ignore
        except Exception:
            SatCVPredictor = None  # type: ignore
        if SatCVPredictor is None:
            # 抛出 Pydantic ValidationError，便于上层统一捕获 SchemaValidationError
            raise ValidationError(
                [
                    {
                        "loc": ("predictor",),
                        "msg": (
                            "predictor=cv_sat 需要可选依赖（如 rasterio/GDAL 等）。"
                            "请安装相应依赖后再试，或将 predictor 设置为 env_nc/dl_ice。"
                        ),
                        "type": "value_error",
                    }
                ],
                RuntimeConfig,
            )
    return cfg


def validate_scenarios_config(data: Dict[str, Any]) -> ScenariosConfig:
    return _model_validate(ScenariosConfig, data)


__all__ = [
    "RuntimeConfig",
    "ScenariosConfig",
    "ScenarioDefaults",
    "ScenarioItem",
    "ValidationError",
    "validate_runtime_config",
    "validate_scenarios_config",
    "model_dump",
    "format_validation_error",
]
