"""
数据页 - 显示环境数据和静态资产状态
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st


def get_manifest_path() -> Path:
    """获取 manifest 路径"""
    # 尝试从环境变量获取
    manifest_env = os.getenv("ARCTICROUTE_MANIFEST")
    if manifest_env:
        return Path(manifest_env)
    
    # 默认路径
    return Path("data_real/manifest.json")


def load_static_assets_doctor() -> dict:
    """加载静态资产检查报告"""
    report_path = Path("reports/static_assets_doctor.json")
    
    if not report_path.exists():
        # 尝试运行 doctor
        try:
            from scripts.static_assets_doctor import check_static_assets
            return check_static_assets()
        except Exception as e:
            return {
                "error": str(e),
                "missing_required": [],
                "missing_optional": [],
                "all_ok": False,
            }
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {
            "error": str(e),
            "missing_required": [],
            "missing_optional": [],
            "all_ok": False,
        }


def scan_static_assets() -> dict:
    """扫描静态资产"""
    assets = {
        "bathymetry": [],
        "ports": [],
        "corridors": [],
        "pub150": [],
        "ais": [],
    }
    
    # 扫描 Bathymetry
    bathymetry_patterns = [
        "data_real/bathymetry/ibcao_v4*.nc",
        "data_real/bathymetry/ibcao_v5*.tif",
        "data_real/bathymetry/*.nc",
        "data_real/bathymetry/*.tif",
    ]
    
    for pattern in bathymetry_patterns:
        for path in Path(".").glob(pattern):
            if path.exists():
                assets["bathymetry"].append({
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "type": path.suffix,
                })
    
    # 扫描 Ports
    ports_patterns = [
        "data_real/ports/*.csv",
        "data_real/ports/*.geojson",
        "data_real/ports/world_port_index*.csv",
    ]
    
    for pattern in ports_patterns:
        for path in Path(".").glob(pattern):
            if path.exists():
                assets["ports"].append({
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "type": path.suffix,
                })
    
    # 扫描 Corridors
    corridors_patterns = [
        "data_real/corridors/*.geojson",
        "data_real/corridors/*.shp",
        "data_real/corridors/*.nc",
    ]
    
    for pattern in corridors_patterns:
        for path in Path(".").glob(pattern):
            if path.exists():
                assets["corridors"].append({
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "type": path.suffix,
                })
    
    # 扫描 Pub150
    pub150_patterns = [
        "data_real/pub150/*.pdf",
        "data_real/rules/*.pdf",
    ]
    
    for pattern in pub150_patterns:
        for path in Path(".").glob(pattern):
            if path.exists():
                assets["pub150"].append({
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "type": path.suffix,
                })
    
    # 扫描 AIS
    ais_patterns = [
        "data_real/ais/derived/*.nc",
        "data_real/ais/*.nc",
    ]
    
    for pattern in ais_patterns:
        for path in Path(".").glob(pattern):
            if path.exists():
                assets["ais"].append({
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "type": path.suffix,
                })
    
    return assets


def render_data() -> None:
    """渲染数据页"""
    
    st.title("🛰️ 数据源状态")
    st.caption("查看环境数据和静态资产的加载状态")
    
    # Manifest 路径
    st.subheader("📋 Manifest 配置")
    
    manifest_path = get_manifest_path()
    manifest_env = os.getenv("ARCTICROUTE_MANIFEST", "未设置")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.text_input(
            "Manifest 路径",
            value=str(manifest_path),
            help="静态资产清单文件路径",
            disabled=True,
        )
    
    with col2:
        if manifest_path.exists():
            st.markdown('<span class="status-badge active">✓ 存在</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge inactive">✗ 缺失</span>', unsafe_allow_html=True)
    
    st.caption(f"环境变量 ARCTICROUTE_MANIFEST: {manifest_env}")
    
    st.markdown("---")
    
    # Static Assets Doctor 摘要
    st.subheader("🔍 静态资产检查 (Doctor)")
    
    doctor_report = load_static_assets_doctor()
    
    if "error" in doctor_report:
        st.error(f"❌ 加载 Doctor 报告失败: {doctor_report['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_req = len(doctor_report.get("missing_required", []))
            st.metric(
                "缺失必需资产",
                missing_req,
                delta="正常" if missing_req == 0 else "异常",
                delta_color="normal" if missing_req == 0 else "inverse",
            )
        
        with col2:
            missing_opt = len(doctor_report.get("missing_optional", []))
            st.metric(
                "缺失可选资产",
                missing_opt,
                delta="可用" if missing_opt < 3 else "部分缺失",
                delta_color="normal" if missing_opt < 3 else "off",
            )
        
        with col3:
            all_ok = doctor_report.get("all_ok", False)
            st.metric(
                "整体状态",
                "✓ 正常" if all_ok else "⚠ 警告",
                delta="所有必需资产已就绪" if all_ok else "存在缺失",
                delta_color="normal" if all_ok else "inverse",
            )
        
        # 显示缺失详情
        if doctor_report.get("missing_required"):
            with st.expander("❌ 缺失的必需资产", expanded=True):
                for asset in doctor_report["missing_required"]:
                    st.text(f"- {asset}")
        
        if doctor_report.get("missing_optional"):
            with st.expander("⚠️ 缺失的可选资产", expanded=False):
                for asset in doctor_report["missing_optional"]:
                    st.text(f"- {asset}")
    
    st.markdown("---")
    
    # 环境数据状态
    st.subheader("🌊 环境数据 (CMEMS)")
    
    # 尝试获取数据目录
    try:
        from arcticroute.core.env import get_newenv_path
        newenv_dir = get_newenv_path()
    except Exception:
        newenv_dir = Path("data/newenv")
    
    cmems_files = {
        "海冰浓度 (SIC)": newenv_dir / "ice_copernicus_sic.nc",
        "海冰厚度 (SIT)": newenv_dir / "ice_copernicus_sit.nc",
        "有效波高 (SWH)": newenv_dir / "wave_swh.nc",
        "海冰漂移 (Drift)": newenv_dir / "ice_drift.nc",
    }
    
    for name, path in cmems_files.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{name}**")
        
        with col2:
            if path.exists():
                st.markdown('<span class="status-badge active">✓ 可用</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge inactive">✗ 缺失</span>', unsafe_allow_html=True)
        
        with col3:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")
            else:
                st.caption("—")
    
    st.markdown("---")
    
    # 静态资产详细状态
    st.subheader("🗺️ 静态资产详情")
    
    if st.button("🔄 重新扫描静态资产", use_container_width=True):
        st.rerun()
    
    assets = scan_static_assets()
    
    # Bathymetry
    with st.expander(f"🏔️ Bathymetry (水深数据) - {len(assets['bathymetry'])} 个文件", expanded=True):
        if assets["bathymetry"]:
            for asset in assets["bathymetry"]:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(asset["path"])
                with col2:
                    st.caption(asset["type"])
                with col3:
                    st.caption(f"{asset['size_mb']:.1f} MB")
        else:
            st.info("未找到 Bathymetry 数据文件")
            st.caption("预期路径: data_real/bathymetry/ibcao_v4*.nc 或 ibcao_v5*.tif")
    
    # Ports
    with st.expander(f"⚓ Ports (港口数据) - {len(assets['ports'])} 个文件", expanded=True):
        if assets["ports"]:
            for asset in assets["ports"]:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(asset["path"])
                with col2:
                    st.caption(asset["type"])
                with col3:
                    st.caption(f"{asset['size_mb']:.1f} MB")
        else:
            st.info("未找到 Ports 数据文件")
            st.caption("预期路径: data_real/ports/world_port_index*.csv")
    
    # Corridors
    with st.expander(f"🛤️ Corridors (航线走廊) - {len(assets['corridors'])} 个文件", expanded=True):
        if assets["corridors"]:
            for asset in assets["corridors"]:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(asset["path"])
                with col2:
                    st.caption(asset["type"])
                with col3:
                    st.caption(f"{asset['size_mb']:.1f} MB")
        else:
            st.info("未找到 Corridors 数据文件")
            st.caption("预期路径: data_real/corridors/*.geojson")
    
    # Pub150
    with st.expander(f"📚 Pub150 规则 - {len(assets['pub150'])} 个文件", expanded=False):
        if assets["pub150"]:
            for asset in assets["pub150"]:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(asset["path"])
                with col2:
                    st.caption(asset["type"])
                with col3:
                    st.caption(f"{asset['size_mb']:.1f} MB")
        else:
            st.info("未找到 Pub150 规则文件")
            st.caption("预期路径: data_real/pub150/*.pdf")
    
    # AIS
    with st.expander(f"🚢 AIS 拥挤度 - {len(assets['ais'])} 个文件", expanded=True):
        if assets["ais"]:
            for asset in assets["ais"]:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(asset["path"])
                with col2:
                    st.caption(asset["type"])
                with col3:
                    st.caption(f"{asset['size_mb']:.1f} MB")
        else:
            st.info("未找到 AIS 拥挤度数据文件")
            st.caption("预期路径: data_real/ais/derived/*.nc")
    
    st.markdown("---")
    
    # 数据配置
    with st.expander("⚙️ 数据配置", expanded=False):
        st.text_input(
            "环境数据目录",
            value=str(newenv_dir),
            help="CMEMS 环境数据存放目录",
        )
        
        st.text_input(
            "静态资产根目录",
            value="data_real",
            help="静态资产（Bathymetry/Ports/Corridors）根目录",
        )
        
        if st.button("保存配置"):
            st.success("✓ 配置已保存（功能开发中）")
        
        if st.button("运行完整 Doctor 检查"):
            try:
                from scripts.static_assets_doctor import check_static_assets
                report = check_static_assets()
                st.success("✓ Doctor 检查完成")
                st.json(report)
            except Exception as e:
                st.error(f"❌ Doctor 检查失败: {e}")
