"""static_assets_doctor.py

医生脚本：验证静态资产清单(manifest)与文件存在性。
要求：
1. 所有必需资产(asset_id)必须出现在 manifest 中且对应文件存在。
2. manifest 中的 missing 列表必须为空。
3. 可选资产目前为空列表，保留接口未来扩展。

执行完毕后会在 reports/static_assets_doctor.json 生成检查报告，
并根据缺失情况返回 0（全部通过）或 1（失败）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 必需的 asset_id 列表
REQUIRED_ASSET_IDS: list[str] = [
    "bathymetry_ibcao_v4_200m_nc",
    "bathymetry_ibcao_v5_1_2025_depth_400m_tif",
    "ports_world_port_index_geojson",
    "corridors_shipping_hydrography_geojson",
    "rules_pub150_pdf",
]

# 可选 asset_id（目前无）
OPTIONAL_ASSET_IDS: list[str] = []

# 默认 manifest 路径
DEFAULT_MANIFEST_PATH = Path("env/static_assets_manifest.json")


def load_manifest() -> Dict[str, Any]:
    """读取 manifest JSON。如果不存在或解析失败则抛出异常。"""
    manifest_env = os.getenv("ARCTICROUTE_STATIC_ASSETS_MANIFEST")
    manifest_path = Path(manifest_env) if manifest_env else DEFAULT_MANIFEST_PATH

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 文件不存在: {manifest_path}")

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"解析 manifest 失败: {manifest_path}: {e}")


def check_assets(manifest: Dict[str, Any]):
    """检查资产是否缺失，返回 (missing_required, missing_optional)"""
    entries: Dict[str, Any] = manifest.get("entries", {})

    missing_required: List[str] = []
    missing_optional: List[str] = []

    # 检查 required
    for aid in REQUIRED_ASSET_IDS:
        info = entries.get(aid)
        if not info:
            missing_required.append(f"{aid} (no manifest entry)")
            continue
        # 检查文件存在
        abspath = info.get("abspath_hint") or info.get("relpath")
        if not abspath:
            missing_required.append(f"{aid} (invalid manifest entry)")
            continue
        if not Path(abspath).exists():
            missing_required.append(f"{aid} -> {abspath} (file missing)")

    # 检查 optional（目前为空但保留逻辑）
    for aid in OPTIONAL_ASSET_IDS:
        info = entries.get(aid)
        if not info:
            missing_optional.append(f"{aid} (no manifest entry)")
            continue
        abspath = info.get("abspath_hint") or info.get("relpath")
        if not abspath or not Path(abspath).exists():
            missing_optional.append(f"{aid} -> {abspath} (file missing)")

    # manifest 自己的 missing 字段也要计入 required 缺失
    manifest_missing: List[str] = manifest.get("missing") or []
    for m in manifest_missing:
        missing_required.append(f"(manifest.missing) {m}")

    # 去重
    missing_required = sorted(set(missing_required))
    missing_optional = sorted(set(missing_optional))

    return missing_required, missing_optional


def write_report(missing_required: List[str], missing_optional: List[str]):
    """写报告 JSON 并打印摘要"""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "static_assets_doctor.json"

    report_data = {
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "all_required_ok": len(missing_required) == 0,
        "all_optional_ok": len(missing_optional) == 0,
    }

    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # console 输出
    print("=== Static Assets Doctor ===")
    print(f"Missing Required: {len(missing_required)}")
    for item in missing_required:
        print(f"  - {item}")
    print(f"Missing Optional: {len(missing_optional)}")
    for item in missing_optional:
        print(f"  - {item}")
    print(f"Report saved to: {report_path}")

    return report_path


def main() -> int:
    try:
        manifest = load_manifest()
    except Exception as e:
        print(f"[static_assets_doctor] 错误: {e}", file=sys.stderr)
        # 如果 manifest 都无法加载，直接视作全部必需资产缺失
        missing_required = ["manifest not found or invalid"]
        missing_optional: List[str] = []
        write_report(missing_required, missing_optional)
        return 1

    missing_required, missing_optional = check_assets(manifest)
    write_report(missing_required, missing_optional)

    # 如果任一缺失则返回非零
    if missing_required or missing_optional:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
