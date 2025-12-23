from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import platform, sys, json
from typing import List

@dataclass
class Check:
    id: str
    level: str   # OK/WARN/FAIL
    title: str
    detail: str
    fix: str

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def run_ui_doctor() -> List[Check]:
    out: List[Check] = []

    out.append(Check(
        id="runtime",
        level="OK",
        title="Runtime",
        detail=f"python={sys.version.split()[0]} platform={platform.system()}",
        fix="—",
    ))

    # cover asset
    cover = Path("arcticroute/ui/assets/arctic_ui_cover.html")
    if _exists(cover):
        out.append(Check("cover", "OK", "Cover asset", str(cover), "—"))
    else:
        out.append(Check("cover", "FAIL", "Cover asset missing", str(cover), "把 arctic_ui_cover.html 放到 arcticroute/ui/assets/ 下"))

    # static assets doctor (optional)
    doctor_json = Path("reports/static_assets_doctor.json")
    if _exists(doctor_json):
        try:
            j = json.loads(doctor_json.read_text(encoding="utf-8"))
            mr = j.get("missing_required", None)
            mo = j.get("missing_optional", None)
            level = "OK" if (mr == 0) else "WARN"
            out.append(Check("static_assets", level, "Static assets doctor", f"missing_required={mr} missing_optional={mo}", "运行 python -m scripts.static_assets_doctor"))
        except Exception as e:
            out.append(Check("static_assets", "WARN", "Static assets doctor parse failed", f"{doctor_json} {e}", "重新运行 python -m scripts.static_assets_doctor"))
    else:
        out.append(Check("static_assets", "WARN", "Static assets doctor not found", "reports/static_assets_doctor.json missing", "运行 python -m scripts.static_assets_doctor"))

    # CMEMS newenv presence
    newenv = Path("data_processed/newenv")
    layers = {
        "sic": newenv / "ice_copernicus_sic.nc",
        "swh": newenv / "wave_swh.nc",
        "sit": newenv / "ice_thickness.nc",
        "drift": newenv / "ice_drift.nc",
    }
    missing = [k for k,p in layers.items() if not _exists(p)]
    if not missing:
        out.append(Check("cmems_newenv", "OK", "CMEMS newenv layers", "sic/swh/sit/drift present", "—"))
    else:
        out.append(Check("cmems_newenv", "WARN", "CMEMS newenv layers", f"missing: {missing}", "运行 python -m scripts.cmems_newenv_sync 或刷新 CMEMS"))

    # AIS density candidates (best-effort)
    try:
        from arcticroute.core.ais_density_select import scan_candidates
        search_dirs = [
            Path("data/ais_density"),
            Path("data_real/ais/density"),
            Path("data_real/ais/derived"),
        ]
        cands = scan_candidates(search_dirs=search_dirs)
        if cands:
            out.append(Check("ais_density", "OK", "AIS density candidates", f"{len(cands)} candidates found", "—"))
        else:
            out.append(Check("ais_density", "WARN", "AIS density candidates", "0 candidates found", "检查 data/ais_density 或 data_real/ais/density 目录是否存在 .nc"))
    except Exception as e:
        out.append(Check("ais_density", "WARN", "AIS density scan unavailable", str(e), "确认 arcticroute.core.ais_density_select 可导入，或先跳过 AIS 密度"))

    # Vessel profiles count
    try:
        from arcticroute.core.eco.vessel_profiles import get_default_profiles
        prof = get_default_profiles()
        out.append(Check("profiles", "OK", "Vessel profiles", f"{len(prof)} profiles in get_default_profiles()", "如需更多船型：新增 get_profile_catalog() 并让 UI 使用它"))
    except Exception as e:
        out.append(Check("profiles", "WARN", "Vessel profiles import failed", str(e), "检查 arcticroute/core/eco/vessel_profiles.py"))

    return out
