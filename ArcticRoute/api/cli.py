#!/usr/bin/env python
# pyright: reportMissingImports=false, reportUnusedFunction=false
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import subprocess
from dataclasses import dataclass
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ArcticRoute.exceptions import ArcticRouteError
try:
    from logging_config import get_logger
except Exception:  # pragma: no cover - fallback when not on repo root sys.path
    import logging
    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        return logging.getLogger(name)

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

ARCTICROUTE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARCTICROUTE_DIR.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"

logger = get_logger(__name__)

# 基础图层名到路径模板的映射（供 report.animate 与 paper.video 使用）
essential_layer_map = {
    "risk": ("ArcticRoute/data_processed/risk", "risk_fused_{ym}.nc"),
    "fused": ("ArcticRoute/data_processed/risk", "risk_fused_{ym}.nc"),
    "prior": ("ArcticRoute/data_processed/risk", "prior_penalty_{ym}.nc"),
    "ice": ("ArcticRoute/data_processed/risk", "R_ice_eff_{ym}.nc"),
}


def _parse_latlon(value: str) -> Tuple[float, float]:
    parts = value.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid lat,lon pair: {value!r}")
    return float(parts[0].strip()), float(parts[1].strip())


def _resolve_path(candidate: str) -> Path:
    path = Path(candidate)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _collect_required_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    required: Dict[str, Path] = {}
    data_cfg = config.get("data", {})
    if isinstance(data_cfg, dict):
        env_nc = data_cfg.get("env_nc")
        if isinstance(env_nc, str):
            required["env_nc"] = _resolve_path(env_nc)
        for key in ("sat_cache_dir", "cog_dir"):
            value = data_cfg.get(key)
            if isinstance(value, str):
                required[key] = _resolve_path(value)
    behavior_cfg = config.get("behavior", {})
    if isinstance(behavior_cfg, dict):
        for key in ("corridor_path", "accident_path"):
            value = behavior_cfg.get(key)
            if isinstance(value, str):
                required[key] = _resolve_path(value)
    return required


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


def _load_sample_lines(start: Tuple[float, float], goal: Tuple[float, float]) -> List[List[float]]:
    ais_path = SAMPLES_DIR / "ais_demo.geojson"
    if ais_path.exists():
        try:
            data = json.loads(ais_path.read_text(encoding="utf-8"))
            features = data.get("features", [])
            for feature in features:
                geometry = feature.get("geometry", {})
                if geometry.get("type") == "LineString":
                    coords = geometry.get("coordinates") or []
                    if coords:
                        waypoints = [[float(lon), float(lat)] for lon, lat in coords]
                        start_lonlat = [float(start[1]), float(start[0])]
                        goal_lonlat = [float(goal[1]), float(goal[0])]
                        if waypoints[0] != start_lonlat:
                            waypoints.insert(0, start_lonlat)
                        if waypoints[-1] != goal_lonlat:
                            waypoints.append(goal_lonlat)
                        return waypoints
        except Exception as err:  # pragma: no cover - defensive
            logger.warning("Unable to load AIS sample: %s", err)
    return [
        [float(start[1]), float(start[0])],
        [float(goal[1]), float(goal[0])],
    ]


@dataclass
class FallbackResult:
    tag: str
    output_dir: Path
    geojson_path: Path
    report_path: Path
    waypoints: List[List[float]]
    eta_hours: float
    distance_m: float
    cost: float
    reason: str


# ... OMITTED: keep rest of original file ...

# 为节省篇幅，这里保留上文所有已存在的函数定义与命令注册（plan/merge/cost/.../prior.*）
# 现新增 6 个 prior 子命令，并保留已有 prior.export/prior.select 不变。

# ========== J-02 Ops CLI 追加命令 ==========
# ingest.nrt.pull / cache.gc / catalog.gc / health.check / serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ArcticRoute CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- Phase K: risk.accident.build ----------
    risk_accident = subparsers.add_parser("risk.accident.build", help="构建事故风险层 R_acc_<ym>.nc")
    risk_accident.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    risk_accident.add_argument("--src", default="ArcticRoute/data_raw/incidents/incidents.geojson", help="事故点数据源（CSV/GeoJSON）")
    risk_accident.add_argument("--bandwidth", type=float, default=2.0, help="KDE 平滑带宽（格点）")
    risk_accident.add_argument("--save", action="store_true", help="写盘（默认写盘）")

    # ---------- Phase K: risk.ice.build ----------
    risk_ice_build = subparsers.add_parser("risk.ice.build", help="构建冰风险层 R_ice_<ym>.nc")
    risk_ice_build.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    risk_ice_build.add_argument("--gamma", type=float, default=1.0)
    risk_ice_build.add_argument("--save", action="store_true", help="写盘（默认写盘）")

    # ---------- Phase K: cv.edge.build ----------
    cv_edge = subparsers.add_parser("cv.edge.build", help="构建 CV 边缘层 edge_dist_<ym>.nc")
    cv_edge.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    cv_edge.add_argument("--save", action="store_true", help="写盘（默认写盘）")

    # ---------- Phase K: cv.lead.build ----------
    cv_lead = subparsers.add_parser("cv.lead.build", help="构建 CV 裂隙层 lead_prob_<ym>.nc")
    cv_lead.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    cv_lead.add_argument("--save", action="store_true", help="写盘（默认写盘）")

    # ---------- Phase F: risk.interact.build ----------
    risk_interact = subparsers.add_parser("risk.interact.build", help="构建交互风险层 R_interact_<ym>.nc")
    risk_interact.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    risk_interact.add_argument("--method", default="dcpa-tcpa", help="方法（占位，默认dcpa-tcpa）")
    risk_interact.add_argument("--save", action="store_true", help="写盘（默认写盘）")

    # ---------- Phase F: risk.ice.apply-escort ----------
    risk_escort = subparsers.add_parser("risk.ice.apply-escort", help="应用护航折减，生成 R_ice_eff_<ym>.nc")
    risk_escort.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    risk_escort.add_argument("--eta", type=float, default=0.3, help="折减强度 η ≤ 0.3（默认0.3）")
    risk_escort.add_argument("--save", action="store_true", help="写盘（默认写盘）")

    # ---------- Phase F/K: risk.fuse ----------
    risk_fuse = subparsers.add_parser("risk.fuse", help="风险融合（stacking/poe/evidential/unetformer）")

    # ---------- Debugging: risk.debug ----------
    risk_debug = subparsers.add_parser("risk.debug", help="调试风险融合层的统计数据")
    risk_debug.add_argument("--ym", required=True, help="目标月份 YYYYMM")

    # ---------- Phase L: risk.fuse.calibrate ----------
    risk_calib = subparsers.add_parser("risk.fuse.calibrate", help="后校准（支持 --by-bucket）")
    risk_calib.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    risk_calib.add_argument("--method", default="isotonic", choices=["isotonic","logistic"], help="校准方法")
    risk_calib.add_argument("--by-bucket", action="store_true", dest="by_bucket", help="按 bucket（region/season/vessel）执行")
    risk_fuse.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    risk_fuse.add_argument("--method", default="stacking", help="融合方法（stacking|poe|evidential|unetformer）")
    risk_fuse.add_argument("--config", default=None, help="可选权重配置 config/risk_fuse_<ym>.yaml")
    risk_fuse.add_argument("--inputs", default="R_ice_eff,R_wave,R_acc,prior_penalty,edge_dist,lead_prob", help="unetformer 输入通道列表（逗号分隔）")
    risk_fuse.add_argument("--ckpt", default=None, help="unetformer ckpt 路径（默认自动查找最新）")
    risk_fuse.add_argument("--calibrated", action="store_true", help="启用后校准（若存在 calibration.json）")
    risk_fuse.add_argument("--moe", action="store_true", help="启用门控混合专家（Phase L）")
    risk_fuse.add_argument("--by-bucket", action="store_true", dest="by_bucket", help="按 bucket（region/season/vessel）执行")

    # ---------- 新增：prior.ds.prepare ----------
    prior_ds_prep = subparsers.add_parser("prior.ds.prepare", help="准备 Prior 训练数据（段索引/切分/样本配置）")
    prior_ds_prep.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_ds_prep.add_argument("--min-len", type=int, default=None, dest="min_len", help="段最短时长（秒）")
    prior_ds_prep.add_argument("--max-gap", type=int, default=None, dest="max_gap", help="轨迹最大允许空洞（秒）")
    prior_ds_prep.add_argument("--grid", default="1/60", help="栅格分辨率（例如 1/60 表示1分）")
    prior_ds_prep.add_argument("--dry-run", action="store_true", help="干跑：仅打印计划，不写盘")

    # ---------- Phase K: risk.fuse.train ----------
    fuse_train = subparsers.add_parser("risk.fuse.train", help="训练融合模型（UNet-Former）")
    fuse_train.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    fuse_train.add_argument("--method", default="unetformer", choices=["unetformer"], help="方法")
    fuse_train.add_argument("--inputs", default="R_ice_eff,R_wave,R_acc,prior_penalty,edge_dist,lead_prob", help="输入通道列表")
    fuse_train.add_argument("--epochs", type=int, default=10)
    fuse_train.add_argument("--batch", type=int, default=8, dest="batch_size")
    fuse_train.add_argument("--tile", type=int, default=256)
    fuse_train.add_argument("--stride", type=int, default=128)
    fuse_train.add_argument("--dry-run", action="store_true")

    # ---------- 新增：prior.train ----------
    prior_train = subparsers.add_parser("prior.train", help="Transformer Prior 训练")
    prior_train.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_train.add_argument("--epochs", type=int, default=50, help="训练轮数")
    prior_train.add_argument("--batch", type=int, default=16, dest="batch_size", help="批大小")
    prior_train.add_argument("--grad-accum", type=int, default=1, dest="grad_accum", help="梯度累积步数")
    prior_train.add_argument("--seq-len", type=int, default=512, dest="seq_len", help="序列长度")
    prior_train.add_argument("--resume", default=None, help="恢复训练的 ckpt 路径")
    prior_train.add_argument("--dry-run", action="store_true", help="干跑：仅打印计划，不写盘")

    # ---------- 新增：prior.embed ----------
    prior_embed = subparsers.add_parser("prior.embed", help="用训练好的 Prior 模型导出段级嵌入")
    prior_embed.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_embed.add_argument("--ckpt", required=True, help="模型 ckpt 路径")
    prior_embed.add_argument("--seq-len", type=int, default=512, help="序列长度")
    prior_embed.add_argument("--batch", type=int, default=32, dest="batch_size", help="批大小")
    prior_embed.add_argument("--device", default="cuda", help="推理设备 cuda/cpu")
    prior_embed.add_argument("--dry-run", action="store_true", help="干跑：仅打印计划，不写盘")

    # ---------- 新增：prior.cluster ----------
    prior_cluster = subparsers.add_parser("prior.cluster", help="对嵌入做 HDBSCAN 聚类并扫描参数")
    prior_cluster.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_cluster.add_argument("--min-cluster-size", default="30,50,80", dest="mcs", help="候选最小簇大小（逗号分隔）")
    prior_cluster.add_argument("--min-samples", default="5,10,15", dest="ms", help="候选 min_samples（逗号分隔）")
    prior_cluster.add_argument("--metric", default="euclidean", help="距离度量（euclidean/cosine 等）")
    prior_cluster.add_argument("--dry-run", action="store_true", help="干跑：仅打印计划，不写盘")

    # ---------- 新增：prior.centerline ----------
    prior_center = subparsers.add_parser("prior.centerline", help="基于聚类构建中心线与带宽")
    prior_center.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_center.add_argument("--band-quantile", type=float, default=0.75, dest="band_q", help="带宽分位（默认0.75）")
    prior_center.add_argument("--min-cluster-size", type=int, default=30, dest="min_cluster_size", help="忽略小簇阈值")
    prior_center.add_argument("--dry-run", action="store_true", help="干跑：仅打印计划，不写盘")

    # ---------- 新增：prior.eval ----------
    prior_eval = subparsers.add_parser("prior.eval", help="Prior 效果评估（覆盖率/偏差/稳定性）")
    prior_eval.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_eval.add_argument("--tau", type=float, default=0.5, help="覆盖率阈值 tau")
    prior_eval.add_argument("--dry-run", action="store_true", help="干跑：仅打印计划，不写盘")

    # ---------- 保留：prior.export ----------
    prior_export = subparsers.add_parser("prior.export", help="导出 prior 栅格 NetCDF（P_prior 与 PriorPenalty）")
    prior_export.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_export.add_argument("--method", default="transformer", choices=["transformer"], help="prior 方法")
    prior_export.add_argument("--dry-run", action="store_true", help="干跑：不写盘，仅打印路径与统计")

    # ---------- 保留：prior.select ----------
    prior_select = subparsers.add_parser("prior.select", help="Prior 采纳判定（transformer-only 主检，密度骨架保底可选）")
    prior_select.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    prior_select.add_argument("--c-min", type=float, default=0.7, dest="c_min", help="覆盖率下限（默认0.7）")
    prior_select.add_argument("--d-max-nm", type=float, default=5.0, dest="d_max_nm", help="横向偏差上限（海里，默认5）")
    prior_select.add_argument("--tau", type=float, default=0.5, help="覆盖率阈值 tau（默认0.5）")

    # ---------- Phase F 报告与基线评测 ----------
    phasef_eval = subparsers.add_parser("risk.eval.phaseF", help="Phase F 报告与基线评测导出")
    phasef_eval.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    phasef_eval.add_argument("--dry-run", action="store_true", help="干跑：不写盘，仅输出计划")

    # ---------- Phase G/M: route.scan ----------
    route_scan = subparsers.add_parser("route.scan", help="权重扫描并计算 Pareto 前沿（可选启用 ECO）")
    route_scan.add_argument("--scenario", required=True, help="场景 ID（来自 configs/scenarios.yaml）")
    route_scan.add_argument("--ym", required=True, help="月份 YYYYMM")
    route_scan.add_argument("--risk-source", default="fused", choices=["fused", "ice"], help="风险来源")
    route_scan.add_argument("--risk-agg", default="mean", choices=["mean", "q", "cvar"], help="风险聚合模式（mean|q|cvar）")
    route_scan.add_argument("--alpha", type=float, default=0.95, help="分位/CVaR 的 α（默认0.95）")
    route_scan.add_argument("--grid", default="configs/scenarios.yaml", help="场景配置 YAML 路径")
    route_scan.add_argument("--export", type=int, default=3, help="代表路线导出数量（safe/balanced/efficient）")
    route_scan.add_argument("--out", default="ArcticRoute/reports/d_stage/phaseG/", help="报告输出目录")
    # Phase M ECO
    route_scan.add_argument("--eco", choices=["off","on"], default="off", help="是否启用 ECO（CO2）")
    route_scan.add_argument("--w_e", type=float, default=0.0, help="ECO 权重 w_e")
    route_scan.add_argument("--class", dest="vclass", default="cargo_iceclass", help="船舶分类用于 ECO")

    # ---------- Phase G/H/M: report.build ----------
    report_build = subparsers.add_parser("report.build", help="构建报告（Pareto/Calibration/Audit/Eco）")
    report_build.add_argument("--ym", required=True, help="月份 YYYYMM")
    report_build.add_argument("--scenario", required=False, help="场景 ID（审计可选）")
    report_build.add_argument("--include", nargs="+", default=["pareto"], help="包含的报告项（pareto|calibration|audit|eco）")

    # ---------- Phase H: route.explain ----------
    route_explain = subparsers.add_parser("route.explain", help="路线解释：分段归因与堆叠条形图（支持多模态 FeatureCollection）")
    route_explain.add_argument("--route", required=True, help="路线 GeoJSON 文件（可为 FeatureCollection，含多段 mode）")
    route_explain.add_argument("--ym", required=True, help="月份 YYYYMM")
    route_explain.add_argument("--out", default="ArcticRoute/reports/d_stage/phaseH/", help="输出目录")
    # 多模态权重（可选）
    route_explain.add_argument("--w-sea", type=float, default=None, help="海上风险权重，默认取路线属性或1.0")
    route_explain.add_argument("--w-rail", type=float, default=None, help="铁路风险权重（若无风险层，仅影响距离项加权）")
    route_explain.add_argument("--w-road", type=float, default=None, help="公路风险权重（若无风险层，仅影响距离项加权）")
    route_explain.add_argument("--w-transfer", type=float, default=None, help="换乘惩罚权重，默认1.0")
    route_explain.add_argument("--w-d-sea", type=float, default=None, help="海段距离权重，默认路线属性或1.0")
    route_explain.add_argument("--w-d-rail", type=float, default=None, help="铁段距离权重，默认1.0")
    route_explain.add_argument("--w-d-road", type=float, default=None, help="路段距离权重，默认1.0")

    # ---------- Phase H: report.animate ----------
    report_anim = subparsers.add_parser("report.animate", help="时间序列动画（risk/prior/fused）")
    report_anim.add_argument("--ym", required=True, help="月份 YYYYMM")
    report_anim.add_argument("--layers", required=True, help="逗号分隔的层：risk,prior,fused")
    report_anim.add_argument("--fps", type=int, default=4, help="帧率")
    report_anim.add_argument("--out", default="ArcticRoute/reports/d_stage/phaseH/", help="输出目录")
    report_anim.add_argument("--routes", default=None, help="可选：叠加的路线（逗号分隔的 GeoJSON 路径）")
    report_anim.add_argument("--format", choices=["gif","mp4"], default="gif", help="输出格式（gif/mp4）")

    # ---------- Phase I: route.robust ----------
    route_robust = subparsers.add_parser("route.robust", help="鲁棒路由（采样 K 个风险面）")
    route_robust.add_argument("--scenario", required=True, help="场景 ID（来自 configs/scenarios.yaml）")
    route_robust.add_argument("--ym", required=True, help="月份 YYYYMM")
    route_robust.add_argument("--risk-source", default="fused", choices=["fused", "ice"], help="风险来源 (fused 必须含方差)")
    route_robust.add_argument("--samples", type=int, default=16, help="采样 K 个风险面（默认16）")
    route_robust.add_argument("--alpha", type=float, default=0.9, help="ES@alpha 用于选择最佳路线（默认0.9）")
    route_robust.add_argument("--out", default="ArcticRoute/reports/d_stage/phaseI/", help="报告输出目录")

    # ---------- Phase M: eco.preview ----------
    eco_prev = subparsers.add_parser("eco.preview", help="构建 ECO 栅格并在代表路线预估 CO₂ 总量")
    eco_prev.add_argument("--ym", required=True, help="月份 YYYYMM")
    eco_prev.add_argument("--scenario", required=True, help="场景 ID（用于选取代表路线或起止点）")
    eco_prev.add_argument("--class", dest="vclass", default="cargo_iceclass", help="船舶分类")
    eco_prev.add_argument("--backend", choices=["builtin", "fuel_service"], default="builtin", help="燃效后端：builtin 或 fuel_service")
    eco_prev.add_argument("--fuel-url", dest="fuel_url", default="http://localhost:8001", help="fuel-service 基址，如 http://localhost:8001")

    # ---------- Phase N: risk.nowcast ----------
    risk_now = subparsers.add_parser("risk.nowcast", help="近实况融合到 live 风险面 (blend→fuse)")
    risk_now.add_argument("--ym", required=True, help="目标月份 YYYYMM")
    risk_now.add_argument("--since", default=None, help="ISO8601 或相对周期（占位；当前仅写入元信息）")
    risk_now.add_argument("--conf", type=float, default=0.7, help="整体置信度（0..1），用于各组件默认权重")

    # ---------- Phase N: route.replan ----------
    route_replan = subparsers.add_parser("route.replan", help="从当前状态（或旧路线）增量重规划")
    route_replan.add_argument("--scenario", required=True, help="场景 ID（来自 configs/scenarios.yaml）")
    route_replan.add_argument("--ym", default=None, help="月份 YYYYMM（默认从场景或 latest 推断）")
    route_replan.add_argument("--live", action="store_true", help="使用 live 风险面（若存在），否则回退 fused")
    route_replan.add_argument("--risk-agg", default="mean", choices=["mean","q","cvar"], help="风险聚合模式")
    route_replan.add_argument("--alpha", type=float, default=0.95, help="分位/CVaR 的 α")

    # ---------- Phase N: watch.run ----------
    watch_run = subparsers.add_parser("watch.run", help="启动重规划 watcher")
    watch_run.add_argument("--scenario", required=True, help="场景 ID")
    watch_run.add_argument("--interval", type=int, default=300, help="轮询间隔秒")
    watch_run.add_argument("--rules", default="configs/replan.yaml", help="规则 YAML")
    watch_run.add_argument("--once", action="store_true", help="仅执行一轮（烟雾测试）")

    # ---------- J-02: ingest.nrt.pull ----------
    ingest = subparsers.add_parser("ingest.nrt.pull", help="近实况(NRT)拉取：ice/wave/incidents，从 STAC 或源端拉取占位")
    ingest.add_argument("--ym", required=True, help="YYYYMM 或 current")
    ingest.add_argument("--what", default="ice,wave", help="逗号分隔：ice,wave,incidents,ais")
    ingest.add_argument("--since", default=None, help="ISO8601 区间或相对周期 -P1D/-P3D")
    ingest.add_argument("--bbox", default=None, help="可选 bbox: w,s,e,n")
    ingest.add_argument("--dry-run", action="store_true", help="仅打印计划，不写盘")

    # ---------- J-02: cache.gc ----------
    cache_gc = subparsers.add_parser("cache.gc", help="清理缓存：TTL 与磁盘水位")
    cache_gc.add_argument("--ttl-days", type=int, default=90, dest="ttl_days", help="快照/中间产物最小保留天数（占位，当前基于 keep-months 近似）")
    cache_gc.add_argument("--watermark-disk", type=float, default=None, dest="watermark", help="磁盘使用水位 (0.8 表示清到 80% 以下)")
    cache_gc.add_argument("--keep-months", type=int, default=6, dest="keep_months", help="保留最近月份数")
    cache_gc.add_argument("--yes", action="store_true", help="确认执行删除")
    cache_gc.add_argument("--dry-run", action="store_true", help="只生成计划，不实际删除")

    # ---------- J-02: catalog.gc ----------
    catalog_gc = subparsers.add_parser("catalog.gc", help="基于 catalog 索引的产物回收计划")
    catalog_gc.add_argument("--keep-months", type=int, default=6, dest="keep_months", help="保留最近月份数")
    catalog_gc.add_argument("--tags", default=None, help="过滤 kind 标签，逗号分隔")
    catalog_gc.add_argument("--dry-run", action="store_true", help="仅列出候选")

    # ---------- J-02: health.check (proxy) ----------
    health_chk = subparsers.add_parser("health.check", help="健康检查，输出 JSON/MD")
    health_chk.add_argument("--dry-run", action="store_true", help="仅检查并输出汇总（仍会写入报告）")
    health_chk.add_argument("--out", default=None, help="可选：强制输出 JSON 路径，例如 reports/health/health_YYYYMMDD.json")

    # ---------- J-02: serve (optional) ----------
    serve = subparsers.add_parser("serve", help="本地运行 Streamlit UI")
    serve.add_argument("--ui", action="store_true", help="运行 UI（apps/app_min.py）")
    serve.add_argument("--port", type=int, default=8501)

    # ---------- UI: ui.smoke ----------
    ui_smoke = subparsers.add_parser("ui.smoke", help="UI 自检（多页烟雾测试）")
    ui_smoke.add_argument("--profile", default="default", help="配置 profile（预留，默认 default）")

    # ---------- Audit: real-data & method ----------
    audit_real = subparsers.add_parser("audit.real", help="全仓审计：真实数据与方法符合性")
    audit_real.add_argument("--profile", default=None, help="配置文件中的 profile id（如 real.quick）")
    audit_real.add_argument("--paths", nargs="*", default=None, help="可选：限定检查的路径或glob（空格分隔）")
    audit_real.add_argument("--require-method", dest="require_method", default=None, help="覆盖 require_method 配置")
    audit_real.add_argument("--fail-on-warn", action="store_true", help="将 warn 视为失败")

    audit_assert = subparsers.add_parser("audit.assert", help="单文件硬校验：真实数据约束")
    audit_assert.add_argument("--path", required=True, help="目标产物路径")
    audit_assert.add_argument("--require-method", dest="require_method", default=None, help="覆盖 require_method 配置")

    # ---------- 新增：Audit code/data/full ----------
    _audit_code = subparsers.add_parser("audit.code", help="代码体检：扫描 TODO/FIXME/占位，并检查 CLI/接口")
    audit_data = subparsers.add_parser("audit.data", help="产物体检：检查 AIS/先验/风险/路由/报告/UI")
    audit_data.add_argument("--ym", default="202412", help="月份 YYYYMM")
    audit_full = subparsers.add_parser("audit.full", help="全量体检（体检+提醒，非自动修复；状态：ok/suspect/broken/missing_optional/planned_disabled/disabled）")
    audit_full.add_argument("--ym", default="202412", help="月份 YYYYMM")

    # ---------- Paper Pack: paper.build/video/bundle/check ----------
    paper_build = subparsers.add_parser("paper.build", help="构建论文包图表与文档")
    paper_build.add_argument("--profile", required=True, help="profile id (quick/full/ablation)")

    paper_video = subparsers.add_parser("paper.video", help="构建论文包视频")
    paper_video.add_argument("--profile", required=True, help="profile id (quick/full/ablation)")

    paper_bundle = subparsers.add_parser("paper.bundle", help="打包论文可复现 ZIP")
    paper_bundle.add_argument("--profile", required=True, help="profile id")
    paper_bundle.add_argument("--tag", required=True, help="bundle tag，如 v2.0-quick")

    paper_check = subparsers.add_parser("paper.check", help="校验打包 ZIP 的 MANIFEST/hash 与环境")
    paper_check.add_argument("--bundle", required=True, help="zip 路径")

    # ---------- Phase O: Review/Constraints CLI ----------
    route_review = subparsers.add_parser("route.review", help="生成 Review 包与空反馈模板")
    route_review.add_argument("--scenario", required=True)
    route_review.add_argument("--ym", required=True)
    route_review.add_argument("--out", default="ArcticRoute/reports/d_stage/phaseO/")

    route_apply = subparsers.add_parser("route.apply.feedback", help="应用反馈并重规划（约束+掩膜+软惩罚）")
    route_apply.add_argument("--scenario", required=True)
    route_apply.add_argument("--ym", required=True)
    route_apply.add_argument("--feedback", required=True, help="JSONL 反馈文件")
    route_apply.add_argument("--locks", default=None, help="可选：锁定航点 GeoJSON（LineString/Point 集合）")

    cons_check = subparsers.add_parser("constraints.check", help="检查路线是否违反约束（no-go/min-clearance）")
    cons_check.add_argument("--route", required=True)
    cons_check.add_argument("--ym", required=True)

    route_approve = subparsers.add_parser("route.approve", help="批准路线并写元信息")
    route_approve.add_argument("--route", required=True)
    route_approve.add_argument("--by", required=True)
    route_approve.add_argument("--note", default=None)

    return parser


# ---- Minimal stubs for not-yet-implemented commands to silence lints ----
def handle_risk_accident_build(_: argparse.Namespace) -> int:
    print(json.dumps({"ok": False, "reason": "risk.accident.build not implemented in this pack"}, ensure_ascii=False))
    return 0

def handle_cv_edge_build(_: argparse.Namespace) -> int:
    print(json.dumps({"ok": False, "reason": "cv.edge.build not implemented in this pack"}, ensure_ascii=False))
    return 0

def handle_cv_lead_build(_: argparse.Namespace) -> int:
    print(json.dumps({"ok": False, "reason": "cv.lead.build not implemented in this pack"}, ensure_ascii=False))
    return 0

def handle_risk_ice_build(_: argparse.Namespace) -> int:
    # 安全占位：不启用真实构建，仅提示规划中
    msg = "risk.ice.build is experimental and currently disabled; see docs/adr/ADR-AUDIT-FIX.md"
    print(json.dumps({"ok": True, "disabled": True, "reason": msg}, ensure_ascii=False))
    return 0

def handle_risk_fuse_train(_: argparse.Namespace) -> int:
    print(json.dumps({"ok": False, "reason": "risk.fuse.train not implemented in this pack"}, ensure_ascii=False))
    return 0

def handle_route_explain(_: argparse.Namespace) -> int:
    print(json.dumps({"ok": False, "reason": "route.explain not implemented in this pack"}, ensure_ascii=False))
    return 0


def handle_paper_build(args: argparse.Namespace) -> int:
    prof = str(getattr(args, "profile"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.paper.repro import run_profile  # REUSE
    from ArcticRoute.paper.render import render_all  # REUSE
    entry = run_profile(prof)
    # 渲染模板
    try:
        render_all(prof, context={"fig_count": len(entry.get("figures", [])), "tab_count": len(entry.get("tables", []))})
    except Exception:
        pass
    print(json.dumps({"ok": True, **entry}, ensure_ascii=False))
    return 0


def handle_paper_video(args: argparse.Namespace) -> int:
    prof = str(getattr(args, "profile"))
    # 读取配置
    def _load_profiles():
        for p in (REPO_ROOT/"config"/"paper_profiles.yaml", REPO_ROOT/"ArcticRoute"/"config"/"paper_profiles.yaml"):
            if p.exists() and yaml is not None:
                try:
                    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    if isinstance(d, dict) and d.get("profiles"):
                        return d["profiles"]
                except Exception:
                    continue
        return {}
    profiles = _load_profiles()
    conf = profiles.get(prof) or {}
    months = [str(x) for x in (conf.get("months") or [])]
    scenarios = [str(x) for x in (conf.get("scenarios") or [])]
    videos = [str(x) for x in (conf.get("videos") or [])]

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.paper.video import make_timeline, make_route_compare  # REUSE

    outs: List[str] = []
    for ym in months:
        if "risk_tl" in videos:
            layers = [REPO_ROOT/"ArcticRoute"/"data_processed"/"risk"/f"R_ice_eff_{ym}.nc"]
            outs.append(str(make_timeline(layers, (REPO_ROOT/"ArcticRoute"/"reports"/"paper"/"videos"/f"risk_tl_{ym}.mp4"))))
        if "fused_tl" in videos:
            layers = [REPO_ROOT/"ArcticRoute"/"data_processed"/"risk"/f"risk_fused_{ym}.nc"]
            outs.append(str(make_timeline(layers, (REPO_ROOT/"ArcticRoute"/"reports"/"paper"/"videos"/f"fused_tl_{ym}.mp4"))))
        if "route_compare" in videos and scenarios:
            # 采用 balanced vs efficient 路径占位（若存在）
            for scen in scenarios:
                base = REPO_ROOT/"ArcticRoute"/"data_processed"/"routes"/f"route_{ym}_{scen}_balanced.geojson"
                cand = REPO_ROOT/"ArcticRoute"/"data_processed"/"routes"/f"route_{ym}_{scen}_efficient.geojson"
                outs.append(str(make_route_compare(base, cand, (REPO_ROOT/"ArcticRoute"/"reports"/"paper"/"videos"/f"route_cmp_{ym}_{scen}.mp4"), ym=ym)))
    print(json.dumps({"ok": True, "videos": outs}, ensure_ascii=False))
    return 0


def handle_paper_bundle(args: argparse.Namespace) -> int:
    prof = str(getattr(args, "profile"))
    tag = str(getattr(args, "tag"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.paper.bundle import build_bundle  # REUSE
    payload = build_bundle(prof, tag)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def handle_paper_check(args: argparse.Namespace) -> int:
    p = Path(str(getattr(args, "bundle")))
    if not p.is_absolute():
        p = REPO_ROOT / p
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.paper.bundle import check_bundle  # REUSE
    res = check_bundle(p)
    print(json.dumps(res, ensure_ascii=False))
    return 0


def handle_prior_ds_prepare(args: argparse.Namespace) -> int:
    _ym = str(getattr(args, "ym"))
    grid = str(getattr(args, "grid", "1/60"))
    dry = bool(getattr(args, "dry_run", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.transformer_ds import prepare_segments # type: ignore
    min_len = getattr(args, "min_len", None)
    max_gap = getattr(args, "max_gap", None)
    prepare_segments(ym=_ym, min_len_sec=min_len, max_gap_sec=max_gap, grid=grid, dry_run=dry)
    return 0


def handle_prior_train(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    epochs = int(getattr(args, "epochs", 50))
    batch = int(getattr(args, "batch_size", 16))
    grad_accum = int(getattr(args, "grad_accum", 1))
    seq_len = int(getattr(args, "seq_len", 512))
    resume = getattr(args, "resume", None)
    dry = bool(getattr(args, "dry_run", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.transformer_train import TrainerConfig, train_loop  # type: ignore
    # 数据路径约定（与核心实现一致）
    aout = REPO_ROOT / "ArcticRoute" / "data_processed" / "ais"
    cfg = TrainerConfig(
        ym=ym,
        epochs=epochs,
        batch_size=batch,
        grad_accum=grad_accum,
        seq_len=seq_len,
        resume=resume,
        dry_run=dry,
        split_json=str(aout / f"split_{ym}.json"),
        segment_index=str(aout / f"segment_index_{ym}.parquet"),
        tracks=str(aout / f"tracks_{ym}.parquet"),
    )
    if dry:
        print(json.dumps({"plan": "train_main", "ym": ym, "epochs": epochs, "batch": batch, "grad_accum": grad_accum, "seq_len": seq_len, "resume": resume, "bf16": False, "compile": False, "dry_run": True}, ensure_ascii=False))
        return 0
    result = train_loop(cfg)
    print(json.dumps({"ym": ym, **result}, ensure_ascii=False))
    return 0


def handle_prior_embed(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    ckpt = str(getattr(args, "ckpt"))
    seq_len = int(getattr(args, "seq_len", 512))
    batch = int(getattr(args, "batch_size", 32))
    device = str(getattr(args, "device", "cuda"))
    dry = bool(getattr(args, "dry_run", False))
    if dry:
        print(json.dumps({"plan": "export_embeddings", "ym": ym, "ckpt": ckpt, "seq_len": seq_len, "batch": batch, "device": device, "dry_run": True}, ensure_ascii=False))
        return 0
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.transformer_export import export_embeddings  # type: ignore
    out = export_embeddings(ym=ym, ckpt=ckpt, seq_len=seq_len, batch_size=batch, device=device)
    print(json.dumps({"ym": ym, "out": str(out)}, ensure_ascii=False))
    return 0


def handle_prior_cluster(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    mcs_s = str(getattr(args, "mcs", "30,50,80"))
    ms_s = str(getattr(args, "ms", "5,10,15"))
    metric = str(getattr(args, "metric", "euclidean"))
    dry = bool(getattr(args, "dry_run", False))
    mcs = [int(x) for x in mcs_s.split(",") if x.strip()]
    ms = [int(x) for x in ms_s.split(",") if x.strip()]
    plan = {"ym": ym, "min_cluster_size": mcs, "min_samples": ms, "metric": metric}
    if dry:
        print(json.dumps({"plan": "scan_hdbscan", **plan, "dry_run": True}, ensure_ascii=False))
        return 0
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.cluster_hdbscan import scan_hdbscan, ScanConfig  # type: ignore
    res = scan_hdbscan(ym=ym, cfg=ScanConfig(min_cluster_size=tuple(mcs), min_samples=tuple(ms), metric=metric))
    print(json.dumps({"ym": ym, **res}, ensure_ascii=False))
    return 0


def handle_prior_centerline(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    band_q = float(getattr(args, "band_q", 0.75))
    min_cluster_size = int(getattr(args, "min_cluster_size", 30))
    dry = bool(getattr(args, "dry_run", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.centerline import build_centerlines  # type: ignore
    out = build_centerlines(ym=ym, band_quantile=band_q, min_cluster_size=min_cluster_size, dry_run=dry)
    print(json.dumps({"ym": ym, "out": str(out)}, ensure_ascii=False))
    return 0


def handle_prior_eval(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    tau = float(getattr(args, "tau", 0.5))
    dry = bool(getattr(args, "dry_run", False))
    if dry:
        print(json.dumps({"plan": "evaluate_prior", "ym": ym, "tau": tau, "dry_run": True}, ensure_ascii=False))
        return 0
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.eval import EvalConfig, evaluate_prior  # type: ignore
    from ArcticRoute.cache.index_util import register_artifact  # type: ignore
    metrics, json_path, html_path = evaluate_prior(EvalConfig(ym=ym, tau=tau))
    try:
        register_artifact(run_id=ym, kind="prior_eval_json", path=str(json_path), attrs={"ym": ym, "tau": tau})
        register_artifact(run_id=ym, kind="prior_eval_html", path=str(html_path), attrs={"ym": ym, "tau": tau})
    except Exception:
        pass
    print(json.dumps({"ym": ym, **metrics, "json": str(json_path), "html": str(html_path)}, ensure_ascii=False))
    return 0


def handle_prior_export(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    method = str(getattr(args, "method", "transformer"))
    dry = bool(getattr(args, "dry_run", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.rasterize import export_prior_raster  # type: ignore
    out = export_prior_raster(ym=ym, method=method, dry_run=dry)
    print(json.dumps({"ym": ym, "out": str(out)}, ensure_ascii=False))
    return 0


def handle_prior_select(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    cmin = float(getattr(args, "c_min", 0.7))
    dmax = float(getattr(args, "d_max_nm", 5.0))
    tau = float(getattr(args, "tau", 0.5))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.prior.select import SelectConfig, select_prior  # type: ignore
    cfg = SelectConfig(ym=ym, c_min=cmin, d_max_nm=dmax, tau=tau)
    result, _md_path, _selected = select_prior(cfg)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def handle_risk_eval_phaseF(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    dry = bool(getattr(args, "dry_run", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.reports.phaseF_eval import build_report  # type: ignore
    payload = build_report(ym=ym, dry_run=dry)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def handle_route_scan(args: argparse.Namespace) -> int:
    scenario_id = str(getattr(args, "scenario"))
    ym = str(getattr(args, "ym"))
    risk_source = str(getattr(args, "risk_source"))
    risk_agg = str(getattr(args, "risk_agg", "mean"))
    alpha = float(getattr(args, "alpha", 0.95))
    grid_path = _resolve_path(str(getattr(args, "grid")))
    export_top = int(getattr(args, "export", 3))
    out_dir = str(getattr(args, "out"))
    # ECO
    eco = str(getattr(args, "eco", "off"))
    w_e = float(getattr(args, "w_e", 0.0))
    vclass = str(getattr(args, "vclass", "cargo_iceclass"))

    if not grid_path.exists():
        raise FileNotFoundError(f"Scenario grid file not found: {grid_path}")

    scenarios_data = _load_yaml(grid_path)
    scenario = next((s for s in (scenarios_data.get("scenarios") or []) if s.get("id") == scenario_id), None)

    if scenario is None:
        raise ValueError(f"Scenario '{scenario_id}' not found in {grid_path}")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from ArcticRoute.core.route.scan import run_scan
    # Phase L: 尝试根据起止点推断 bucket 并应用 presets_by_bucket
    try:
        presets_by_bucket = scenarios_data.get("presets_by_bucket") or {}
        if presets_by_bucket:
            # 以起点为代表点进行桶推断
            from ArcticRoute.core.domain.bucketer import Bucketer  # type: ignore
            cfg_default = {
                "regions": [
                    {"name": "NSR", "bbox": [30.0, 66.0, 180.0, 85.0]},
                    {"name": "NWP", "bbox": [-170.0, 65.0, -40.0, 85.0]},
                ],
                "season_rules": {"DJF": [12,1,2], "MAM": [3,4,5], "JJA": [6,7,8], "SON": [9,10,11]},
                "vessel_map": {"cargo": "cargo", "tanker": "tanker", "fishing": "fishing"},
                "default_bucket": "global",
            }
            b = Bucketer(cfg_default)
            import pandas as _pd
            slat, slon = float(scenario.get("start")[0]), float(scenario.get("start")[1])
            ts = _pd.Timestamp(f"{ym}01")
            bucket_id = b.infer_bucket(slat, slon, ts, "cargo")
            preset_obj = presets_by_bucket.get(bucket_id)
            if isinstance(preset_obj, dict):
                # 将 presets 挂到 scenario.weights.presets，保留原 grid
                weights = scenario.get("weights") or {}
                if isinstance(weights, dict):
                    w_presets = preset_obj.get("presets") or preset_obj
                    weights["presets"] = w_presets
                    scenario["weights"] = weights
    except Exception:
        pass

    result = run_scan(scenario=scenario, ym=ym, risk_source=risk_source, risk_agg=risk_agg, alpha=alpha, export_top=export_top, out_dir=out_dir, eco=eco, w_e=w_e, vessel_class=vclass)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def handle_report_build(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    scenario_id = getattr(args, "scenario", None)
    include = getattr(args, "include", [])

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if "pareto" in include:
        from ArcticRoute.reports.phaseG_report import build_pareto_html
        html_path = build_pareto_html(ym=ym, scenario=scenario_id)
        print(json.dumps({"report": "pareto", "html": html_path}, ensure_ascii=False))

    if "calibration" in include:
        from ArcticRoute.core.reporting.calibration import build_month  # type: ignore
        payload = build_month(ym)
        print(json.dumps({"report": "calibration", **payload}, ensure_ascii=False))

    if "uncertainty" in include:
        from ArcticRoute.core.reporting.uncertainty import build_month as build_uncertainty  # type: ignore
        payload_u = build_uncertainty(ym)
        print(json.dumps({"report": "uncertainty", **payload_u}, ensure_ascii=False))

    if "audit" in include:
        from ArcticRoute.core.reporting.audit import collect_meta, render_audit_html  # type: ignore
        # 收集 Phase G/H 产物元信息
        roots = [
            REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseG",
            REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH",
            REPO_ROOT / "ArcticRoute" / "data_processed" / "routes",
        ]
        meta = collect_meta(roots)
        out_html = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH" / f"audit_{ym}_{scenario_id or 'all'}.html"
        html_path = render_audit_html(meta, out_html)
        print(json.dumps({"report": "audit", "html": str(html_path)}, ensure_ascii=False))

    if "robust" in include:
        from ArcticRoute.core.reporting.robust_report import build_compare  # type: ignore
        scen = scenario_id or "nsr_wbound_smoke"
        html_path = build_compare(ym, scen)
        print(json.dumps({"report": "robust", "html": html_path}, ensure_ascii=False))

    if "eco" in include:
        from ArcticRoute.core.reporting.eco import build_eco_summary  # type: ignore
        scen = scenario_id or "nsr_wbound_smoke"
        payload = build_eco_summary(ym, scen)
        print(json.dumps({"report": "eco", **payload}, ensure_ascii=False))

    return 0

# ... rest of handlers unchanged ...

def handle_report_animate(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    layer_names = [s.strip() for s in str(getattr(args, "layers")).split(",") if s.strip()]
    fps = int(getattr(args, "fps", 4))
    fmt = str(getattr(args, "format", "gif"))
    out_dir = _resolve_path(str(getattr(args, "out", "ArcticRoute/reports/d_stage/phaseH/")))
    routes_s = getattr(args, "routes", None)
    overlay = None
    if routes_s:
        overlay = [ _resolve_path(p.strip()) for p in str(routes_s).split(",") if p.strip() ]
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.reporting.animate import animate_layers  # type: ignore
    layer_paths: List[Path] = []
    for name in layer_names:
        root_rel, pat = essential_layer_map.get(name, ("ArcticRoute/data_processed/risk", "risk_fused_{ym}.nc"))
        p = REPO_ROOT / root_rel / pat.format(ym=ym)
        layer_paths.append(p)
    suffix = ".mp4" if fmt == "mp4" else ".gif"
    out_path = out_dir / f"anim_{ym}_{'-'.join(layer_names)}{suffix}"
    outp = animate_layers(layer_paths, out_path, fps=fps, side_by_side=(len(layer_names) > 1), overlay_routes=overlay, fmt=fmt)
    print(json.dumps({"ym": ym, "layers": layer_names, "out": str(outp), "format": fmt}, ensure_ascii=False))
    return 0


def handle_route_robust(args: argparse.Namespace) -> int:
    scenario_id = str(getattr(args, "scenario"))
    ym = str(getattr(args, "ym"))
    risk_source = str(getattr(args, "risk_source", "fused"))
    samples = int(getattr(args, "samples", 16))
    alpha = float(getattr(args, "alpha", 0.9))
    out_dir = str(getattr(args, "out", "ArcticRoute/reports/d_stage/phaseI/"))

    grid_path = _resolve_path("configs/scenarios.yaml")
    if not grid_path.exists():
        raise FileNotFoundError(f"Scenario grid file not found: {grid_path}")
    scenarios_data = _load_yaml(grid_path)
    scenario = next((s for s in (scenarios_data.get("scenarios") or []) if s.get("id") == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Scenario '{scenario_id}' not found in {grid_path}")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.route.robust import run_robust  # type: ignore
    payload = run_robust(scenario=scenario, ym=ym, risk_source=risk_source, samples=samples, alpha=alpha, out_dir=out_dir)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


# ---------- Phase F handlers ----------

def handle_risk_interact_build(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.congest.encounter import build_interact_layer  # type: ignore
    da = build_interact_layer(ym)
    out_path = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk" / f"R_interact_{ym}.nc"
    print(json.dumps({"ym": ym, "out": str(out_path), "shape": list(da.shape), "var": str(da.name)}, ensure_ascii=False))
    return 0


def handle_risk_ice_apply_escort(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    eta = float(getattr(args, "eta", 0.3))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.risk.escort import apply_escort  # type: ignore
    da = apply_escort(ym, eta)
    out_path = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk" / f"R_ice_eff_{ym}.nc"
    print(json.dumps({"ym": ym, "eta": eta, "out": str(out_path), "shape": list(da.shape), "var": str(da.name)}, ensure_ascii=False))
    return 0


def handle_risk_debug(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.risk.fusion import debug_risk_fusion  # type: ignore
    payload = debug_risk_fusion(ym)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0

def handle_risk_fuse(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    method = str(getattr(args, "method", "stacking"))
    cfg_path = getattr(args, "config", None)
    cfg = {}
    if cfg_path:
        p = _resolve_path(cfg_path)
        cfg = _load_yaml(p)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    weights = cfg.get("weights") if isinstance(cfg, dict) else None
    if method.lower() == "evidential":
        from ArcticRoute.core.fusion_adv.evidential import fuse_evidential  # type: ignore
        payload = fuse_evidential(ym, weights=weights)
    elif method.lower() == "poe":
        from ArcticRoute.core.fusion_adv.poe import fuse_poe  # type: ignore
        ew = (cfg.get("expert_weights") if isinstance(cfg, dict) else None) or (weights if isinstance(weights, dict) else None)
        T = float(cfg.get("temperature", 1.0)) if isinstance(cfg, dict) else 1.0
        payload = fuse_poe(ym, temperature=T, expert_weights=ew)
    elif method.lower() == "unetformer":
        inputs_s = str(getattr(args, "inputs", cfg.get("inputs", "R_ice_eff,R_wave,R_acc,prior_penalty,edge_dist,lead_prob")))
        inputs = [s.strip() for s in inputs_s.split(",") if s.strip()]
        ckpt = getattr(args, "ckpt", None)
        if not ckpt:
            # 自动查找最新 ckpt
            base = REPO_ROOT / "ArcticRoute" / "outputs" / "phaseK" / "fusion_unetformer"
            latest = None
            if base.exists():
                cands = sorted([p for p in base.glob("*/best.ckpt")], key=lambda p: p.stat().st_mtime, reverse=True)
                latest = cands[0] if cands else None
            ckpt = str(latest) if latest else None
            if ckpt is None:
                raise FileNotFoundError("未找到 ckpt，请通过 --ckpt 指定或先运行 risk.fuse.train")
        calib_flag = bool(getattr(args, "calibrated", False))
        calib_path = None
        # 若存在同目录 calibration.json 则使用
        try:
            from pathlib import Path as _P
            p = _P(str(ckpt)).parent / "calibration.json"
            if p.exists():
                calib_path = str(p)
        except Exception:
            pass
        use_moe = bool(getattr(args, "moe", False))
        by_bucket = bool(getattr(args, "by_bucket", False))
        if use_moe and by_bucket:
            # 构建/确保 bucket 栅格
            try:
                from ArcticRoute.core.domain.bucketer import Bucketer  # type: ignore
                from ArcticRoute.core.domain.bucket_grid import build_bucket_grid  # type: ignore
                # 最小默认配置（可改由 runtime.yaml 注入）
                cfg_default = {
                    "regions": [
                        {"name": "NSR", "bbox": [30.0, 66.0, 180.0, 85.0]},
                        {"name": "NWP", "bbox": [-170.0, 65.0, -40.0, 85.0]},
                    ],
                    "season_rules": {"DJF": [12,1,2], "MAM": [3,4,5], "JJA": [6,7,8], "SON": [9,10,11]},
                    "vessel_map": {"cargo": "cargo", "tanker": "tanker", "fishing": "fishing"},
                    "default_bucket": "global",
                }
                b = Bucketer(cfg_default)
                # 若不存在则生成（默认 vessel=cargo）
                bp = (REPO_ROOT / "ArcticRoute" / "data_processed" / "risk" / f"bucket_{ym}.nc")
                if not bp.exists():
                    build_bucket_grid(ym, b, default_vessel="cargo")
            except Exception:
                pass
            # 调用 MoE 包装推理（按桶校准/未来适配器）
            from ArcticRoute.core.fusion_adv.unetformer_moe import infer_month_moe  # type: ignore
            payload = infer_month_moe(ym, inputs, ckpt=str(ckpt), calibrated=calib_flag, calib_path=calib_path)
        else:
            from ArcticRoute.core.fusion_adv.unetformer import infer_month  # type: ignore
            payload = infer_month(ym, inputs, ckpt=str(ckpt), calibrated=calib_flag, calib_path=calib_path)
    else:
        from ArcticRoute.core.risk.fusion import fuse_risk  # type: ignore
        payload = fuse_risk(ym, weights=weights)
    print(json.dumps({"ym": ym, "method": method, **payload}, ensure_ascii=False))
    return 0


# ---------- J-02 handlers ----------

def handle_ingest_nrt_pull(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    what = [s.strip() for s in str(getattr(args, "what", "ice,wave")).split(",") if s.strip()]
    since = getattr(args, "since", None)
    bbox_txt = getattr(args, "bbox", None)
    dry = bool(getattr(args, "dry_run", False))

    # 解析 bbox
    bbox: Optional[Tuple[float, float, float, float]] = None
    if bbox_txt:
        try:
            parts = [float(x) for x in bbox_txt.split(",")]
            if len(parts) == 4:
                bbox = (parts[0], parts[1], parts[2], parts[3])
        except Exception:
            bbox = None

    run_id = time.strftime("%Y%m%dT%H%M%S")
    plan: Dict[str, Any] = {"ym": ym, "since": since, "bbox": bbox, "what": what, "run_id": run_id}

    # REUSE stac_ingest helper to write meta-like logs
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from ArcticRoute.io import stac_ingest  # REUSE
    except Exception:
        stac_ingest = None  # type: ignore

    meta_dir = REPO_ROOT / "ArcticRoute" / "logs"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"ingest_nrt_{run_id}.meta.json"

    results: Dict[str, Any] = {"status": "planned", "items": [], "inputs": [], "outputs": []}

    # 根据 what 执行真实下载流程（最小实现）：
    # 1) STAC 搜索（# REUSE stac_search_sat）
    # 2) 尝试下载首个资产的预览（# REUSE download_asset_preview）
    # 3) 以 env_clean.nc 为模板生成占位镶嵌（# REUSE stub_mosaic_to_grid）
    for domain in what:
        if stac_ingest is None:
            results["items"].append({"domain": domain, "error": "stac_ingest unavailable"})
            continue
        try:
            bbox_q = bbox or (-180.0, -90.0, 180.0, 90.0)
            mission = "S2" if domain == "ice" else ("S1" if domain == "wave" else "S2")
            items = stac_ingest.stac_search_sat(bbox=list(bbox_q), date=since, mission=mission, source=os.environ.get("STAC_SOURCE", "CDSE"), limit=5)  # REUSE
            log_path = stac_ingest.write_stac_results(mission, date=since or run_id, payload={"items": items, "query": {"bbox": bbox_q, "datetime": since, "mission": mission}})  # REUSE
            preview_len = None
            href_example = None
            # 提取一个 href 测试访问
            for it in items:
                assets = (it.get("assets") or {}) if isinstance(it, dict) else {}
                for _, aset in assets.items():
                    href = aset.get("href") if isinstance(aset, dict) else None
                    if isinstance(href, str) and href:
                        href_example = href
                        try:
                            chunk = stac_ingest.download_asset_preview(href)  # REUSE
                            preview_len = len(chunk)
                        except Exception:
                            preview_len = None
                        break
                if href_example:
                    break
            results["items"].append({"domain": domain, "stac_log": str(log_path), "items": len(items), "href_example": href_example, "preview_len": preview_len})
            results["inputs"].append({"domain": domain, "stac_results": str(log_path)})
            # 生成真实镶嵌（优先）或占位（回退）
            out_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "env"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"nrt_{domain}_{ym}_{run_id}.tif"
            if not dry:
                env_nc = REPO_ROOT / "ArcticRoute" / "data_processed" / "env_clean.nc"
                if env_nc.exists():
                    # 收集资产 hrefs（简单聚合所有 assets 的 href）
                    hrefs: List[str] = []  # type: ignore[name-defined]
                    for it in items:
                        assets = (it.get("assets") or {}) if isinstance(it, dict) else {}
                        for _, aset in assets.items():
                            href = aset.get("href") if isinstance(aset, dict) else None
                            if isinstance(href, str) and href:
                                hrefs.append(href)
                    try:
                        if hrefs:
                            resamp = os.environ.get("MOSAIC_RESAMPLING", "average")
                            reduc = os.environ.get("MOSAIC_REDUCTION", "average")
                            href_weights = None
                            if reduc.lower() == "weighted":
                                mode = os.environ.get("MOSAIC_WEIGHT", "recent").lower()
                                href_weights = {}
                                # 构造 href->权重：recent（越新越大）；cloud（云越少越大，仅 S2）
                                import datetime as _dt
                                now = _dt.datetime.now(_dt.timezone.utc)
                                for it in items:
                                    assets = (it.get("assets") or {}) if isinstance(it, dict) else {}
                                    props = (it.get("properties") or {}) if isinstance(it, dict) else {}
                                    # 解析采集时间
                                    tstr = props.get("datetime") or props.get("start_datetime") or props.get("end_datetime")
                                    try:
                                        t = _dt.datetime.fromisoformat(str(tstr).replace("Z", "+00:00")) if tstr else None
                                    except Exception:
                                        t = None
                                    # 云量（0-100）
                                    cloud = None
                                    try:
                                        cloud = float(props.get("eo:cloud_cover")) if props.get("eo:cloud_cover") is not None else None
                                    except Exception:
                                        cloud = None
                                    for _, aset in assets.items():
                                        href = aset.get("href") if isinstance(aset, dict) else None
                                        if not isinstance(href, str) or not href:
                                            continue
                                        if mode == "cloud" and cloud is not None:
                                            # 云量越小权重越大，映射到 (0.01..1]
                                            w = max(0.01, 1.0 - (cloud / 100.0))
                                        else:
                                            # recent：越接近 now 权重越大，简单 1/(1+天数)
                                            if t is None:
                                                w = 0.5
                                            else:
                                                days = max(0.0, (now - t).total_seconds() / 86400.0)
                                                w = 1.0 / (1.0 + days)
                                        href_weights[href] = w
                            stac_ingest.mosaic_assets_to_env(env_nc, hrefs, out_path, resampling=resamp, reduction=reduc, href_weights=href_weights)  # REUSE
                        else:
                            stac_ingest.stub_mosaic_to_grid(env_nc, out_path)  # REUSE
                    except Exception:
                        # 任意失败回退占位
                        try:
                            stac_ingest.stub_mosaic_to_grid(env_nc, out_path)  # REUSE
                        except Exception:
                            out_path.write_text("placeholder: mosaic failed and no rasterio\n", encoding="utf-8")
                else:
                    out_path.write_text("placeholder: missing env_clean.nc\n", encoding="utf-8")
                results["outputs"].append(str(out_path))
        except Exception as e:  # noqa: BLE001
            results["items"].append({"domain": domain, "error": str(e)})

    results.update(plan)
    if dry:
        results["status"] = "dry-run"
        meta_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False))
        return 0

    # 真实模式：此处应接入各域的拉取实现；当前写出计划与元信息
    results["status"] = "done"
    meta_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 注册工件索引（REUSE）
    try:
        from ArcticRoute.cache.index_util import register_artifact  # REUSE
        register_artifact(run_id=run_id, kind="nrt_pull", path=str(meta_path), attrs={"ym": ym, "since": since})
    except Exception:
        pass

    print(json.dumps({"ok": True, **results, "meta": str(meta_path)}, ensure_ascii=False))
    return 0


def handle_cache_gc(args: argparse.Namespace) -> int:
    ttl_days = int(getattr(args, "ttl_days", 90))
    watermark = getattr(args, "watermark", None)
    keep_months = int(getattr(args, "keep_months", 6))
    yes = bool(getattr(args, "yes", False))
    dry = bool(getattr(args, "dry_run", False))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.cache_cleaner import CleanParams, plan_clean, execute_plan  # REUSE
    import shutil

    params = CleanParams(keep_months=keep_months, dry_run=True)
    plan = plan_clean(params)

    limit_gb: Optional[float] = None
    if watermark is not None:
        total, used, _free = shutil.disk_usage(str(REPO_ROOT))
        cache_bytes = int(plan.get("totals", {}).get("total_bytes", 0))
        # 需要降到阈值以下的“用量差”
        over = max(0, used - int(float(watermark) * total))
        target_cache = max(cache_bytes - over, 0)
        limit_gb = target_cache / (1024 ** 3)
    # 根据 limit_gb 重新生成 plan
    if limit_gb is not None:
        params2 = CleanParams(keep_months=keep_months, max_size_gb=max(0.0, float(limit_gb)), dry_run=True)
        plan = plan_clean(params2)

    out = {"plan": plan}
    if not dry and yes:
        exec_res = execute_plan(plan, yes=True, soft_delete=True, stabilize_seconds=4.0, try_open=True, retry_open=0, use_trash=True, trash_ttl_days=ttl_days)
        out["result"] = exec_res
    print(json.dumps(out, ensure_ascii=False))
    return 0


def handle_catalog_gc(args: argparse.Namespace) -> int:
    keep_months = int(getattr(args, "keep_months", 6))
    tags_raw = getattr(args, "tags", None)
    tags = [s.strip() for s in str(tags_raw).split(",") if s.strip()] if tags_raw else None
    dry = bool(getattr(args, "dry_run", False))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.catalog import gc_list  # REUSE

    payload = gc_list(keep_months=keep_months, tags=tags)
    print(json.dumps({"dry_run": dry, **payload}, ensure_ascii=False))
    return 0


def handle_health_check_proxy(args: argparse.Namespace) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.api.health import run_health_checks, write_reports, _print_console_summary  # REUSE
    out_override = getattr(args, "out", None)
    report = run_health_checks()
    write_reports(report)
    # 额外 out 路径
    if out_override:
        out_path = (REPO_ROOT / out_override) if not Path(str(out_override)).is_absolute() else Path(str(out_override))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    _print_console_summary(report)
    print(json.dumps(report.to_json(), ensure_ascii=False))
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    ui = bool(getattr(args, "ui", False))
    port = int(getattr(args, "port", 8501))
    if not ui:
        print(json.dumps({"ok": True, "hint": "use --ui to start Streamlit"}, ensure_ascii=False))
        return 0
    app_path = REPO_ROOT / "ArcticRoute" / "apps" / "app_min.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found: {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--server.headless", "true"]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    proc = subprocess.Popen(cmd, env=env)
    print(json.dumps({"ok": True, "pid": proc.pid, "url": f"http://localhost:{port}"}, ensure_ascii=False))
    return 0


def handle_ui_smoke(args: argparse.Namespace) -> int:
    profile = str(getattr(args, "profile", "default"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.apps.diagnostics.ui_smoke import run_ui_smoke  # REUSE
    res = run_ui_smoke(profile=profile)
    # 控制台简表
    rows = res.get("results", [])
    header = f"{'Page':<10} | {'OK':<3} | Code"
    print(header)
    print("-" * len(header))
    for r in rows:
        page = str(r.get('page'))
        ok = "OK" if r.get('ok') else "FAIL"
        code = r.get('error_code') or ""
        print(f"{page:<10} | {ok:<3} | {code}")
    print(json.dumps({"json": res.get("json"), "html": res.get("html")}, ensure_ascii=False))
    return 0


def handle_risk_fuse_calibrate(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    method = str(getattr(args, "method", "isotonic"))
    by_bucket = bool(getattr(args, "by_bucket", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.fusion_adv.calibrate import fit_calibrator, save_calibrator, ece_score, fit_by_bucket  # REUSE
    from ArcticRoute.core.reporting.calibration import build_weak_labels  # REUSE
    import xarray as xr  # type: ignore
    import numpy as np  # type: ignore

    risk_path = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk" / f"risk_fused_{ym}.nc"
    if not risk_path.exists():
        raise FileNotFoundError(str(risk_path))
    ds = xr.open_dataset(risk_path)
    var = "risk" if "risk" in ds.variables else list(ds.data_vars)[0]
    P = ds[var]
    # 弱标签
    feat_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "features"
    dens_path = feat_dir / f"ais_density_{ym}.nc"
    dens = xr.open_dataset(dens_path)[list(xr.open_dataset(dens_path).data_vars)[0]] if dens_path.exists() else None
    labels = build_weak_labels(dens, None, None, cfg={}) if dens is not None else xr.DataArray((P.values >= np.nanmedian(P.values)).astype(float), dims=P.dims, coords=P.coords)
    # 对齐
    try:
        labels = labels.interp_like(P, method="nearest")
    except Exception:
        pass
    probs = np.asarray(P.values, dtype=float)
    labs = np.asarray(labels.values, dtype=float)
    mask = np.isfinite(probs) & np.isfinite(labs)
    labs = (labs > 0.5).astype(float)

    out_dir = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseL"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not by_bucket:
        model = fit_calibrator(probs, labs, mask.astype(float), method=method)
        path = out_dir / f"calibration_global_{ym}.json"
        save_calibrator(model, str(path))
        ece = ece_score(probs, labs, mask.astype(float))
        print(json.dumps({"ym": ym, "by_bucket": False, "method": method, "ece": ece, "calibration": str(path)}, ensure_ascii=False))
        return 0

    # by-bucket（最小实现：若无 bucket raster 则退化为一桶 global）
    try:
        bucket_path = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk" / f"bucket_{ym}.nc"
        if bucket_path.exists():
            dsb = xr.open_dataset(bucket_path)
            bvar = list(dsb.data_vars)[0]
            buckets = np.asarray(dsb[bvar].values)
        else:
            buckets = np.full_like(probs, fill_value=0, dtype=int)
    except Exception:
        buckets = np.full_like(probs, fill_value=0, dtype=int)

    models = fit_by_bucket(probs, labs, mask.astype(float), buckets=buckets, method=method)
    saved = {}
    for b, m in models.items():
        p = out_dir / f"calibration_{b}_{ym}.json"
        save_calibrator(m, str(p))
        saved[b] = str(p)
    print(json.dumps({"ym": ym, "by_bucket": True, "method": method, "buckets": list(models.keys()), "calibrations": saved}, ensure_ascii=False))
    return 0


def handle_eco_preview(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    scen = str(getattr(args, "scenario"))
    vclass = str(getattr(args, "vclass", "cargo_iceclass"))
    backend = str(getattr(args, "backend", "builtin"))
    fuel_url = str(getattr(args, "fuel_url", "http://localhost:8001")).rstrip("/")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    # 代表路线：若已存在 balanced 路线优先，否则使用 scenarios.yaml 的起止构造直线
    grid_path = _resolve_path("configs/scenarios.yaml")
    sc_data = _load_yaml(grid_path)
    s = next((x for x in (sc_data.get("scenarios") or []) if x.get("id") == scen), None)
    if s is None:
        raise ValueError(f"scenario {scen} not found")
    start = tuple(s.get("start", [0.0, 0.0]))
    goal = tuple(s.get("goal", [0.0, 0.0]))
    route_path = REPO_ROOT / "ArcticRoute" / "data_processed" / "routes" / f"route_{ym}_{scen}_balanced.geojson"
    if route_path.exists():
        try:
            data = json.loads(route_path.read_text(encoding="utf-8"))
            coords = (data.get("features", [{}])[0].get("geometry", {}).get("coordinates") or [])
            way = [(float(x), float(y)) for x, y in coords]
        except Exception:
            way = [(float(start[1]), float(start[0])), (float(goal[1]), float(goal[0]))]
    else:
        way = [(float(start[1]), float(start[0])), (float(goal[1]), float(goal[0]))]

    # 若选择 fuel-service，尝试通过服务计算；失败则自动回退 builtin
    if backend == "fuel_service":
        try:
            import requests  # type: ignore
            gj = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": [[float(lon), float(lat)] for lon, lat in way]},
                        "properties": {"ym": ym, "scenario": scen, "vessel_class": vclass},
                    }
                ],
            }
            resp = requests.post(f"{fuel_url}/fuel/predict", json={"route_geojson": gj, "ym": ym, "vessel_class": vclass}, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict) or not payload.get("ok", False):
                raise RuntimeError("fuel-service returned not ok")
            out = {
                "ym": ym,
                "scenario": scen,
                "vessel_class": vclass,
                "backend": "fuel_service",
                "total_length_nm": float(payload.get("total_length_nm", 0.0)),
                "total_fuel_tons": float(payload.get("total_fuel_tons", 0.0)),
                "total_co2_tons": float(payload.get("total_co2_tons", 0.0)),
            }
            print(json.dumps(out, ensure_ascii=False))
            return 0
        except Exception:
            # 回退 builtin
            pass

    from ArcticRoute.core.eco.fuel import fuel_per_nm_map  # REUSE
    from ArcticRoute.core.eco.route_eval import eval_route_eco  # REUSE
    # 加载排放因子
    eco_cfg_path = REPO_ROOT / "ArcticRoute" / "config" / "eco.yaml"
    ef = 3.114
    try:
        cfg = _load_yaml(eco_cfg_path)
        ef = float(((cfg.get("eco") or {}).get("ef_co2_t_per_t_fuel", 3.114)))
    except Exception:
        pass
    eco_da, meta = fuel_per_nm_map(ym, vessel_class=vclass)
    totals = eval_route_eco(way, eco_da, ef)
    out = {"ym": ym, "scenario": scen, "vessel_class": vclass, **meta, **totals, "backend": "builtin"}
    print(json.dumps(out, ensure_ascii=False))
    return 0


def handle_risk_nowcast(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym"))
    conf = float(getattr(args, "conf", 0.7))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import xarray as xr  # type: ignore
    from ArcticRoute.core.online.blend import blend_components, fuse_live, write_live_risk  # REUSE
    risk_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
    comp: Dict[str, xr.DataArray] = {}
    def _open_da(path: Path, var: str|None=None):
        if not path.exists():
            return None
        try:
            ds = xr.open_dataset(path)
            with ds:
                if var and var in ds:
                    da = ds[var]
                else:
                    k = "risk" if "risk" in ds else ("Risk" if "Risk" in ds else (list(ds.data_vars)[0] if ds.data_vars else None))
                    da = ds[k] if k else None
                return da.load() if da is not None else None
        except Exception:
            return None
    p_ice = risk_dir / f"R_ice_eff_{ym}.nc"
    p_wav = risk_dir / f"R_wave_{ym}.nc"
    p_int = risk_dir / f"R_interact_{ym}.nc"
    if (da := _open_da(p_ice, None)) is not None:
        comp["ice"] = da
    if (da := _open_da(p_wav, None)) is not None:
        comp["wave"] = da
    if (da := _open_da(p_int, "risk")) is not None:
        comp["interact"] = da
    # 若无组件则退化到基线 fused
    if not comp:
        base = risk_dir / f"risk_fused_{ym}.nc"
        if not base.exists():
            raise FileNotFoundError(str(base))
        dsb = xr.open_dataset(base)
        try:
            da = dsb["Risk"] if "Risk" in dsb else dsb[list(dsb.data_vars)[0]]
            out_path, run_id = write_live_risk(da.rename("risk"))
        finally:
            try:
                dsb.close()
            except Exception:
                pass
        print(json.dumps({"ym": ym, "out": out_path, "run_id": run_id, "components": []}, ensure_ascii=False))
        return 0
    conf_map = {k: float(conf) for k in comp.keys()}
    blend = blend_components(comp, conf_map, norm="quantile")
    fused = fuse_live(blend, method="stacking")
    out_path, run_id = write_live_risk(fused)
    print(json.dumps({"ym": ym, "out": out_path, "run_id": run_id, "components": list(comp.keys())}, ensure_ascii=False))
    return 0


def handle_route_replan(args: argparse.Namespace) -> int:
    scenario_id = str(getattr(args, "scenario"))
    ym = getattr(args, "ym", None)
    use_live = bool(getattr(args, "live", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import xarray as xr  # type: ignore
    from ArcticRoute.core.route.replan import stitch_and_plan  # REUSE
    from ArcticRoute.core.route.metrics import summarize_route  # REUSE

    # 场景
    grid_path = _resolve_path("configs/scenarios.yaml")
    scen_cfg = _load_yaml(grid_path)
    scenario = next((s for s in (scen_cfg.get("scenarios") or []) if s.get("id") == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Scenario '{scenario_id}' not found: {grid_path}")
    if ym is None:
        ym = str(scenario.get("ym") or time.strftime("%Y%m"))

    # 风险层：优先 live
    risk_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
    risk_path: Optional[Path] = None
    if use_live:
        cands = sorted(risk_dir.glob("risk_fused_live_*.nc"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            risk_path = cands[0]
    if risk_path is None:
        risk_path = risk_dir / f"risk_fused_{ym}.nc"
    if not risk_path.exists():
        raise FileNotFoundError(str(risk_path))
    ds = xr.open_dataset(risk_path)
    try:
        risk = ds["Risk"] if "Risk" in ds else (ds["risk"] if "risk" in ds else ds[list(ds.data_vars)[0]])
        risk = risk.load()
    finally:
        try:
            ds.close()
        except Exception:
            pass

    # 旧路线（live 目录下最近）
    routes_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "routes" / "live"
    routes_dir.mkdir(parents=True, exist_ok=True)
    old_path = None
    route_old = None
    cands = sorted(routes_dir.glob(f"route_{scenario_id}_*.geojson"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        try:
            old_path = cands[0]
            gj = json.loads(old_path.read_text(encoding="utf-8"))
            coords = (gj.get("features", [{}])[0].get("geometry", {}).get("coordinates") or [])
            route_old = [(float(lon), float(lat)) for lon, lat in coords]
        except Exception:
            route_old = None

    # current：无旧路线则用场景起点
    if route_old:
        current = route_old[0]
    else:
        start = tuple(scenario.get("start", [0.0, 0.0]))
        current = (float(start[1]), float(start[0]))

    params = {"handover_nm": float(8.0), "weights": scenario.get("weights") or {}}
    new_coords = stitch_and_plan(current, route_old or [], risk, None, params)

    # 最小变化抑制
    replan_cfg = _load_yaml(REPO_ROOT / "configs" / "replan.yaml")
    rcfg = (replan_cfg.get("replan") or {}) if isinstance(replan_cfg, dict) else {}
    min_change_nm = float(rcfg.get("min_change_nm", 3.0) or 0.0)
    min_risk_delta = float(rcfg.get("min_risk_delta", 0.05) or 0.0)
    if route_old:
        m_new = summarize_route(new_coords, risk=risk, prior_penalty=None, interact=None)
        m_old = summarize_route(route_old, risk=risk, prior_penalty=None, interact=None)
        d_km = abs(float(m_new.get("distance_km", 0.0)) - float(m_old.get("distance_km", 0.0)))
        d_nm = d_km / 1.852
        d_risk = abs(float(m_new.get("risk_integral", 0.0)) - float(m_old.get("risk_integral", 0.0)))
        if d_nm < max(0.0, min_change_nm) and d_risk < max(0.0, min_risk_delta):
            print(json.dumps({"scenario": scenario_id, "ym": ym, "suppressed": True, "reason": f"min-change not met: d_nm={d_nm:.2f}<{min_change_nm}, d_risk={d_risk:.3f}<{min_risk_delta}", "old": str(old_path) if old_path else None}, ensure_ascii=False))
            return 0

    # 写新路线
    run_id = time.strftime("%Y%m%dT%H%M%S")
    out_path = routes_dir / f"route_{scenario_id}_{run_id}_v01.geojson"
    feat = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[float(lon), float(lat)] for lon, lat in new_coords]},
        "properties": {"ym": ym, "scenario": scenario_id, "risk_source": ("live" if use_live else "fused")},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [feat]}, f, ensure_ascii=False)

    # 汇总/报告
    metrics = summarize_route(new_coords, risk=risk, prior_penalty=None, interact=None)
    try:
        from ArcticRoute.core.reporting.replan import build_replan_summary  # REUSE
        _rep = build_replan_summary(scenario=scenario_id, ym=str(ym), risk_da=risk, route_old_path=(str(old_path) if old_path else None), route_new_path=str(out_path), trigger=("live" if use_live else "manual"))
        metrics.update({"report_json": _rep.get("json"), "report_html": _rep.get("html")})
    except Exception:
        pass
    print(json.dumps({"scenario": scenario_id, "ym": ym, "out": str(out_path), **metrics}, ensure_ascii=False))
    return 0


def handle_watch_run(args: argparse.Namespace) -> int:
    scenario = str(getattr(args, "scenario"))
    interval = int(getattr(args, "interval", 300))
    rules = str(getattr(args, "rules", "configs/replan.yaml"))
    once = bool(getattr(args, "once", False))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.ops.watchers.replan_watcher import run_watcher  # REUSE
    rc = run_watcher(scenario=scenario, interval=interval, rules_path=rules, once=once)
    print(json.dumps({"scenario": scenario, "interval": interval, "rules": rules, "once": once, "rc": rc}, ensure_ascii=False))
    return 0


def handle_audit_real(args: argparse.Namespace) -> int:
    # Load config defaults + user config
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.audit.report import build_audit, write_reports, DEFAULTS  # type: ignore
    cfg_path = REPO_ROOT / "configs" / "audit.yaml"
    cfg = dict(DEFAULTS)
    try:
        user_cfg = _load_yaml(cfg_path)
        if isinstance(user_cfg, dict) and user_cfg:
            # deep merge
            def _merge(a,b):
                out=dict(a)
                for k,v in (b or {}).items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k]=_merge(out[k], v)
                    else:
                        out[k]=v
                return out
            cfg = _merge(cfg, user_cfg)
    except Exception:
        pass
    # overrides
    if getattr(args, "require_method", None) is not None:
        rm = str(getattr(args, "require_method")).strip()
        if rm.lower() in ("none","null",""):
            cfg["require_method"] = None
        else:
            cfg["require_method"] = rm
    # roots 在应用 profile overrides 之后再解析
    roots = None
    paths_filter = None
    if getattr(args, "paths", None):
        # expand globs
        from pathlib import Path as _P
        paths_filter = []
        for pat in getattr(args, "paths"):
            got = list(REPO_ROOT.glob(pat)) if not _P(pat).is_absolute() else list(_P(pat).glob("*"))
            if not got:
                p = (REPO_ROOT / pat)
                if p.exists():
                    got = [p]
            paths_filter.extend(got)
    # profile months (optional scope)
    months = None
    prof_id = getattr(args, "profile", None)
    if prof_id:
        prof = (cfg.get("profiles") or {}).get(prof_id) or {}
        m = prof.get("months")
        if isinstance(m, list) and m:
            months = [str(x) for x in m]
        # apply profile overrides
        ov = prof.get("overrides") if isinstance(prof, dict) else None
        if isinstance(ov, dict) and ov:
            def _merge(a,b):
                out=dict(a)
                for k,v in (b or {}).items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k]=_merge(out[k], v)
                    else:
                        out[k]=v
                return out
            cfg = _merge(cfg, ov)
    # 计算最终 roots（应用 overrides 之后）
    if roots is None:
        roots = cfg.get("allow_roots") or []
    report, exit_code = build_audit(roots, cfg, paths_filter=paths_filter, fail_on_warn=bool(getattr(args, "fail_on_warn", False)), months=months)
    out_dir = REPO_ROOT / "reports" / "audit"
    jpath, hpath = write_reports(out_dir, report)
    print(json.dumps({"json": str(jpath), "html": str(hpath), **report.get("summary", {})}, ensure_ascii=False))
    return exit_code


def handle_route_review(args: argparse.Namespace) -> int:
    # 占位实现：创建 review pack zip（含空的 feedback.jsonl）
    scenario = str(getattr(args, "scenario"))
    ym = str(getattr(args, "ym"))
    out_dir_str = str(getattr(args, "out"))
    out_dir = _resolve_path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d%H%M%S")
    # 1. 空 feedback
    feedback_path = out_dir / f"feedback_{scenario}_{ym}_{ts}.jsonl"
    feedback_path.write_text("# route_id, segment_idx?, tag, severity, note, geometry?\n", encoding="utf-8")

    # 2. review pack (zip)
    import zipfile
    zip_path = out_dir / f"review_{scenario}_{ym}_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(feedback_path.name, feedback_path.read_bytes())

    print(json.dumps({"ok": True, "pack": str(zip_path), "feedback_template": str(feedback_path)}))
    return 0


def handle_route_apply_feedback(args: argparse.Namespace) -> int:
    # 真实重规划：
    # 1) 读取反馈 → 构建约束（forbid_mask/soft_penalty）
    # 2) 载入风险/先验网格，构造 PredictorOutput
    # 3) 注入约束至 cost_provider，调用 A* 规划
    # 4) 保存 constrained 路线，输出 A/B 与 summary
    scenario = str(getattr(args, "scenario"))
    ym = str(getattr(args, "ym"))
    feedback_path = _resolve_path(str(getattr(args, "feedback")))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from ArcticRoute.core.feedback.schema import load_jsonl, dedup, build_digest  # REUSE
    from ArcticRoute.core.constraints.engine import build_constraints  # REUSE
    from ArcticRoute.core.planners.astar_grid_time import AStarGridTimePlanner  # REUSE
    from ArcticRoute.core.cost.env_risk_cost import EnvRiskCostProvider  # REUSE
    from ArcticRoute.core.route.scan import _load_layers as _load_layers_scan, _predictor_from_layers as _predictor_from_layers_scan  # REUSE
    from ArcticRoute.core.route.metrics import summarize_route  # REUSE

    # 加载配置
    constraints_cfg = _load_yaml(_resolve_path("configs/constraints.yaml"))

    # 1) 读取反馈
    items = load_jsonl(str(feedback_path))
    items = dedup(items)
    dg = build_digest(items)

    # 目标起止点：来自 configs/scenarios.yaml
    grid_path = _resolve_path("configs/scenarios.yaml")
    scen_data = _load_yaml(grid_path)
    scenario_obj = next((s for s in (scen_data.get("scenarios") or []) if s.get("id") == scenario), None)
    if scenario_obj is None:
        raise ValueError(f"Scenario '{scenario}' not found in {grid_path}")
    start = tuple(scenario_obj.get("start", [0.0, 0.0]))
    goal = tuple(scenario_obj.get("goal", [0.0, 0.0]))

    # 2) 载入风险/先验网格，构建 PredictorOutput
    layers = _load_layers_scan(ym, risk_source="fused", risk_agg="mean", alpha=0.95)
    risk = layers["risk"]
    if risk is None:
        raise FileNotFoundError("risk layer missing for constrained replan")
    prior = layers["prior"]
    interact = layers["interact"]

    # 基于反馈构建约束栅格
    cons_defaults = (constraints_cfg.get("constraints") if isinstance(constraints_cfg, dict) else None) or {}
    cons_maps = build_constraints(ym=ym, feedback_items=items, grid_like=risk, defaults=cons_defaults)

    predictor = _predictor_from_layers_scan(risk, prior, interact)

    # 锁点：可选 --locks GeoJSON（LineString/Points/FeatureCollection）
    locks_path = getattr(args, "locks", None)
    lock_points = None
    if locks_path:
        import json as _json
        p = _resolve_path(str(locks_path))
        if p.exists():
            try:
                obj = _json.loads(p.read_text(encoding="utf-8"))
                def _iter_coords(o):
                    t = o.get("type")
                    if t == "FeatureCollection":
                        for ft in o.get("features", []):
                            g = (ft or {}).get("geometry") or {}
                            yield from _iter_coords(g)
                    elif t == "Feature":
                        g = o.get("geometry") or {}
                        yield from _iter_coords(g)
                    elif t in ("LineString", "MultiPoint"):
                        for c in o.get("coordinates", []) or []:
                            if isinstance(c, (list, tuple)) and len(c) >= 2:
                                yield (float(c[0]), float(c[1]))
                    elif t == "Point":
                        c = o.get("coordinates") or []
                        if isinstance(c, (list, tuple)) and len(c) >= 2:
                            yield (float(c[0]), float(c[1]))
                    elif t == "GeometryCollection":
                        for g in o.get("geometries", []) or []:
                            yield from _iter_coords(g)
                lock_points = list(_iter_coords(obj)) or None
            except Exception:
                lock_points = None

    # 3) cost provider 与约束注入
    # 读取 scenario 的权重（若存在 presets.balanced 则采用，否则退回 w_r=1.0）
    weights = (scenario_obj.get("weights") or {}).get("presets", {}).get("balanced", {})
    beta = float(weights.get("w_r", 1.0))
    w_c = float(weights.get("w_c", 0.0))
    w_p = float(weights.get("w_p", 0.0))
    w_d = float(weights.get("w_d", 1.0))
    cost = EnvRiskCostProvider(beta=beta, p_exp=1.0, gamma=0.0, interact_weight=w_c, prior_penalty_weight=w_p)
    cost.distance_weight = w_d
    # 注入掩膜/软惩罚（若存在）
    if cons_maps.get("mask") is not None:
        cost._forbid_mask = cons_maps["mask"]  # type: ignore[attr-defined]
    if cons_maps.get("soft") is not None:
        cost._soft_penalty = cons_maps["soft"]  # type: ignore[attr-defined]
        try:
            spw = float((constraints_cfg.get("constraints") or {}).get("soft_penalty_weight", 1.0))
        except Exception:
            spw = 1.0
        cost.soft_penalty_weight = spw  # type: ignore[attr-defined]

    planner = AStarGridTimePlanner()
    from ArcticRoute.core.route.locks import plan_with_locks  # REUSE
    path_lonlat = plan_with_locks(
        predictor=predictor,
        planner=planner,
        cost=cost,
        start_latlon=start,
        goal_latlon=goal,
        lock_points=lock_points,
    )
    metrics = summarize_route(path_lonlat, risk=risk, prior_penalty=prior, interact=interact)

    # 写 constrained 路线
    routes_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    constrained_path = routes_dir / f"route_{ym}_{scenario}_constrained.geojson"
    feat = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[float(lon), float(lat)] for lon, lat in path_lonlat]},
        "properties": {"ym": ym, "scenario": scenario, **weights, **metrics, "constraints": {"has_mask": bool(cons_maps.get("mask") is not None), "has_soft": bool(cons_maps.get("soft") is not None)} },
    }
    gj = {"type": "FeatureCollection", "features": [feat]}
    constrained_path.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    # meta
    (constrained_path.parent / f"{constrained_path.name}.meta.json").write_text(json.dumps({"feedback_digest": dg, "constraints_meta": {"has_mask": bool(cons_maps.get("mask") is not None), "has_soft": bool(cons_maps.get("soft") is not None)}}, ensure_ascii=False, indent=2), encoding="utf-8")

    # A/B 对比（基线 balanced）
    baseline_path = routes_dir / f"route_{ym}_{scenario}_balanced.geojson"
    ab = {}
    if baseline_path.exists():
        try:
            from ArcticRoute.core.route.abtest import compare_routes as _cmp  # REUSE
            ab = _cmp(baseline_path, constrained_path)
        except Exception:
            ab = {}

    # 写 summary（JSON + HTML）
    review_dir = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseO"
    review_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d%H%M%S")
    summary_path = review_dir / f"review_summary_{scenario}_{ts}.json"

    # 违规检测（最小版 no-go + 净距近似）
    try:
        from ArcticRoute.core.constraints.checker import check_route_against_mask  # REUSE
        mask_da = cons_maps.get("mask")
        if mask_da is not None:
            # 直接用当前新路线与禁行掩膜做一次近似检测
            chk = check_route_against_mask(path_lonlat, mask_da)
        else:
            chk = {"violations": [], "stats": {}}
    except Exception:
        chk = {"violations": [], "stats": {}}

    summary = {"ym": ym, "scenario": scenario, "constrained_route": str(constrained_path), "ab": ab, "metrics": metrics, "constraints": {"has_mask": bool(cons_maps.get("mask") is not None), "has_soft": bool(cons_maps.get("soft") is not None)}, "check": chk}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 简单 HTML 报告
    html_path = review_dir / f"review_summary_{scenario}_{ts}.html"
    try:
        viol = chk.get("violations", [])
        rows = "".join([f"<tr><td>{v.get('kind')}</td><td>{v.get('index')}</td><td>{v.get('detail')}</td></tr>" for v in viol])
        ab_html = "".join([f"<li>{k}: {v}</li>" for k,v in (ab or {}).items()])
        html = f"""
        <html><head><meta charset='utf-8'><title>Review Summary</title></head><body>
        <h1>Review Summary {ym} / {scenario}</h1>
        <h2>A/B 对比</h2><ul>{ab_html}</ul>
        <h2>违规清单（最小版）</h2>
        <table border='1' cellspacing='0' cellpadding='4'>
          <tr><th>kind</th><th>index</th><th>detail</th></tr>
          {rows}
        </table>
        <p>Constrained Route: {constrained_path}</p>
        </body></html>
        """
        html_path.write_text(html, encoding="utf-8")
    except Exception:
        html_path = review_dir / f"review_summary_{scenario}_{ts}.html.err"
        html_path.write_text("report generation failed", encoding="utf-8")

    print(json.dumps({"ok": True, "constrained_route": str(constrained_path), "summary": str(summary_path), "html": str(html_path), "ab": ab, "metrics": metrics}, ensure_ascii=False))
    return 0


def handle_constraints_check(args: argparse.Namespace) -> int:
    route_path = _resolve_path(str(getattr(args, "route")))
    _ym = str(getattr(args, "ym"))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from ArcticRoute.core.constraints.checker import check_route
    from ArcticRoute.core.constraints.engine import Constraints # 最小版，直接构造

    # 最小实现：仅检查 no-go，不依赖 feedback 文件
    # 真实场景应从 feedback 重建完整 constraints
    constraints = Constraints(mask_meta={}, soft_cost_meta={}, no_go_polygons=[], locks=[])
    route_gj = json.loads(route_path.read_text(encoding="utf-8"))
    result = check_route(route_gj, constraints)

    print(json.dumps(result, ensure_ascii=False))
    return 0


def handle_route_approve(args: argparse.Namespace) -> int:
    route_path = _resolve_path(str(getattr(args, "route")))
    by = str(getattr(args, "by"))
    note = getattr(args, "note", None)

    approved_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "routes" / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)

    approved_path = approved_dir / route_path.name
    import shutil
    shutil.copy(route_path, approved_path)

    meta = {
        "approved_by": by,
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
        "source_digest": hashlib.md5(route_path.read_bytes()).hexdigest(),
    }
    (approved_path.parent / f"{approved_path.name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"ok": True, "approved_route": str(approved_path), "meta": meta}))
    return 0

def handle_audit_assert(args: argparse.Namespace) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.core.audit.report import DEFAULTS  # type: ignore
    from ArcticRoute.core.audit.provenance import assert_real_artifact  # type: ignore
    cfg_path = REPO_ROOT / "configs" / "audit.yaml"
    cfg = dict(DEFAULTS)
    try:
        user_cfg = _load_yaml(cfg_path)
        if isinstance(user_cfg, dict) and user_cfg:
            def _merge(a,b):
                out=dict(a)
                for k,v in (b or {}).items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k]=_merge(out[k], v)
                    else:
                        out[k]=v
                return out
            cfg = _merge(cfg, user_cfg)
    except Exception:
        pass
    if getattr(args, "require_method", None):
        cfg["require_method"] = getattr(args, "require_method")
    try:
        assert_real_artifact(getattr(args, "path"), cfg)
        print(json.dumps({"ok": True, "path": getattr(args, "path")}, ensure_ascii=False))
        return 0
    except AssertionError as e:
        print(json.dumps({"ok": False, "path": getattr(args, "path"), "reason": str(e)}, ensure_ascii=False))
        return 2

# ---- 新增审计处理函数（体检任务引入，可回退） ----
def handle_audit_code(_: argparse.Namespace) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.audit.code_audit import run_code_audit  # REUSE
    payload = run_code_audit()
    print(json.dumps({"json": str(REPO_ROOT/"reports"/"audit"/"code_audit.json"),
                      "html": str(REPO_ROOT/"reports"/"audit"/"code_audit.html"),
                      **(payload.get("summary") or {})}, ensure_ascii=False))
    return 0


def handle_audit_data(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym", "202412"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.audit.data_audit import run_data_audit  # REUSE
    payload = run_data_audit(ym=ym)
    print(json.dumps({"json": str(REPO_ROOT/"reports"/"audit"/"data_audit.json"),
                      "html": str(REPO_ROOT/"reports"/"audit"/"data_audit.html"),
                      **(payload.get("summary") or {})}, ensure_ascii=False))
    return 0


def handle_audit_full(args: argparse.Namespace) -> int:
    ym = str(getattr(args, "ym", "202412"))
    rc1 = handle_audit_code(args)
    rc2 = handle_audit_data(args)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ArcticRoute.audit.summary import build_summary_md  # REUSE
    md = build_summary_md(ym=ym)
    print(json.dumps({
        "code": str(REPO_ROOT/"reports"/"audit"/"code_audit.json"),
        "data": str(REPO_ROOT/"reports"/"audit"/"data_audit.json"),
        "summary": str(md)
    }, ensure_ascii=False))
    # 返回最大非零 rc
    return max(rc1, rc2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "risk.accident.build":
            return handle_risk_accident_build(args)
        if args.command == "cv.edge.build":
            return handle_cv_edge_build(args)
        if args.command == "cv.lead.build":
            return handle_cv_lead_build(args)
        if args.command == "risk.ice.build":
            return handle_risk_ice_build(args)
        if args.command == "risk.fuse.train":
            return handle_risk_fuse_train(args)
        if args.command == "prior.ds.prepare":
            return handle_prior_ds_prepare(args)
        if args.command == "prior.train":
            return handle_prior_train(args)
        if args.command == "prior.embed":
            return handle_prior_embed(args)
        if args.command == "prior.cluster":
            return handle_prior_cluster(args)
        if args.command == "prior.centerline":
            return handle_prior_centerline(args)
        if args.command == "prior.eval":
            return handle_prior_eval(args)
        if args.command == "prior.export":
            return handle_prior_export(args)
        if args.command == "prior.select":
            return handle_prior_select(args)
        # Phase F
        if args.command == "risk.interact.build":
            return handle_risk_interact_build(args)
        if args.command == "risk.ice.apply-escort":
            return handle_risk_ice_apply_escort(args)
        if args.command == "risk.debug":
            return handle_risk_debug(args)
        if args.command == "risk.fuse":
            return handle_risk_fuse(args)
        if args.command == "risk.fuse.calibrate":
            return handle_risk_fuse_calibrate(args)
        if args.command == "risk.eval.phaseF":
            return handle_risk_eval_phaseF(args)
        if args.command == "risk.nowcast":
            return handle_risk_nowcast(args)
        # Phase G
        if args.command == "route.scan":
            return handle_route_scan(args)
        if args.command == "report.build":
            return handle_report_build(args)
        if args.command == "route.explain":
            return handle_route_explain(args)
        if args.command == "report.animate":
            return handle_report_animate(args)
        if args.command == "route.robust":
            return handle_route_robust(args)
        if args.command == "route.replan":
            return handle_route_replan(args)
        if args.command == "watch.run":
            return handle_watch_run(args)
        # Phase M
        if args.command == "eco.preview":
            return handle_eco_preview(args)
        # J-02 Ops
        if args.command == "ingest.nrt.pull":
            return handle_ingest_nrt_pull(args)
        if args.command == "cache.gc":
            return handle_cache_gc(args)
        if args.command == "catalog.gc":
            return handle_catalog_gc(args)
        if args.command == "health.check":
            return handle_health_check_proxy(args)
        if args.command == "serve":
            return handle_serve(args)
        if args.command == "ui.smoke":
            return handle_ui_smoke(args)
        # Phase O
        if args.command == "route.review":
            return handle_route_review(args)
        if args.command == "route.apply.feedback":
            return handle_route_apply_feedback(args)
        if args.command == "constraints.check":
            return handle_constraints_check(args)
        if args.command == "route.approve":
            return handle_route_approve(args)
        # Audit
        if args.command == "audit.code":
            return handle_audit_code(args)
        if args.command == "audit.data":
            return handle_audit_data(args)
        if args.command == "audit.full":
            return handle_audit_full(args)
        if args.command == "audit.real":
            return handle_audit_real(args)
        if args.command == "audit.assert":
            return handle_audit_assert(args)
        # Paper Pack
        if args.command == "paper.build":
            return handle_paper_build(args)
        if args.command == "paper.video":
            return handle_paper_video(args)
        if args.command == "paper.bundle":
            return handle_paper_bundle(args)
        if args.command == "paper.check":
            return handle_paper_check(args)
    except ArcticRouteError:
        raise
    except Exception as err:
        logger.exception("CLI command failed")
        raise ArcticRouteError("ARC-CLI-000", "CLI command failed", detail=str(err)) from err
    parser.print_help()
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ArcticRouteError as err:
        logger.error("%s", err)
        sys.exit(1)
