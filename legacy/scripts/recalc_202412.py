from __future__ import annotations
import os, json, glob, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 1) Build risk_ice_202412 with cv_cache edge fusion (function already enhanced)
print("[STEP] risk_ice_202412: start")
from ArcticRoute.core.risk.ice import build_risk_ice
pi = build_risk_ice(ym="202412", t0=0.0, t1=1.0, gamma=1.0, dry_run=False)
print(json.dumps({"risk_ice": pi}, ensure_ascii=False))

# 2) Accident: build Q4 window GeoJSON and run accident KDE
print("[STEP] risk_accident_202412: prepare incidents Q4")
import pandas as pd
inc_pq = REPO / "ArcticRoute" / "data_processed" / "incidents" / "incidents_clean.parquet"
if not inc_pq.exists():
    raise SystemExit(f"incidents_clean.parquet not found: {inc_pq}")
df = pd.read_parquet(inc_pq)[["time_utc","lat","lon"]].dropna()
df["ym"] = df["time_utc"].astype(str).str.slice(0,7).str.replace('-', '')
df_q4 = df[df["ym"].isin(["202410","202411","202412"])]
outs = REPO / "outputs"
outs.mkdir(parents=True, exist_ok=True)
inc_gj = outs / "incidents_Q4_2024.geojson"
feats = [{"type":"Feature","geometry":{"type":"Point","coordinates":[float(r.lon), float(r.lat)]},"properties":{"ts": str(r.time_utc)}} for r in df_q4.itertuples(index=False)]
inc_gj.write_text(json.dumps({"type":"FeatureCollection","features":feats}, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"incidents_geojson": str(inc_gj), "count": len(feats)}, ensure_ascii=False))

print("[STEP] risk_accident_202412: build")
from ArcticRoute.core.risk.accident import build_risk_accident
pa = build_risk_accident(ym="202412", acc_src=str(inc_gj), bandwidth_cells=2.0, dry_run=False)
print(json.dumps({"risk_accident": pa}, ensure_ascii=False))

# 3) Fuse with unetformer + calibration
print("[STEP] risk_fused_202412: fuse (unetformer)")
base = REPO / "ArcticRoute" / "outputs" / "phaseK" / "fusion_unetformer"
ckpt = None
for p in sorted(base.glob("*/best.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True):
    ckpt = str(p)
    break
if ckpt is None:
    raise SystemExit("no unetformer ckpt found under outputs/phaseK/fusion_unetformer/*/best.ckpt")
calib_path = str(Path(ckpt).parent / "calibration.json")
from ArcticRoute.core.fusion_adv.unetformer import infer_month
# Inputs list同 CLI 默认
inputs = ["R_ice_eff","R_wave","R_acc","prior_penalty","edge_dist","lead_prob"]
res = infer_month("202412", inputs, ckpt=ckpt, calibrated=True, calib_path=(calib_path if Path(calib_path).exists() else None))
print(json.dumps({"risk_fused": res}, ensure_ascii=False))

print("[DONE] recalc 202412 completed")






