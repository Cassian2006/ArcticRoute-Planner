"""
数据页 - 显示环境数据和静态资产状态
集成数据发现功能，提供可解释的数据源搜索
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st

from arcticroute.io.data_discovery import (
    discover_cmems_layers,
    clear_discovery_caches,
    get_cmems_status_summary,
    DEFAULT_AIS_DIRS,
    DEFAULT_NEWENV_DIRS,
    DEFAULT_CACHE_DIRS,
)
from arcticroute.core.ais_density_select import scan_candidates, select_best_candidate
from arcticroute.core.cost import compute_grid_signature
from arcticroute.core.grid import load_grid_with_landmask


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
        return {
            "error": "Report not found. Click 'Re-scan' to generate.",
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


def run_static_assets_doctor() -> dict:
    """运行静态资产检查脚本"""
    try:
        result = subprocess.run(
            ["python", "-m", "scripts.static_assets_doctor"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # 尝试加载生成的报告
        report = load_static_assets_doctor()
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "report": report,
            "timestamp": datetime.now().isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "error": "Timeout: Doctor script took too long",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def run_cmems_newenv_sync() -> dict:
    """?? CMEMS ??????"""

    try:
        result = subprocess.run(
            ["python", "-m", "scripts.cmems_newenv_sync"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "error": "Timeout: sync script took too long",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def scan_static_assets() -> dict:
    """???????????? UI ??/??????"""
    candidates = {
        "bathymetry": Path("data_real/bathymetry.nc"),
        "ports": Path("data_real/ports"),
        "corridors": Path("data_real/corridors"),
        "ais": Path("data_real/ais"),
    }
    return {key: path.exists() for key, path in candidates.items()}


def render_data() -> None:
    """渲染数据页"""
    
    st.title(" 数据源状态")
    st.caption("查看环境数据和静态资产的加载状态")
    
    # ========== CMEMS 环境数据 ==========
    st.subheader(" CMEMS 环境数据定位")
    
    # 数据定位按钮
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(" 重新扫描 CMEMS 数据", use_container_width=True):
            st.toast("开始同步 CMEMS 缓存并重新扫描...")
            clear_discovery_caches()
            st.cache_data.clear()
            with st.spinner("正在同步 CMEMS 缓存到 newenv..."):
                sync_result = run_cmems_newenv_sync()
            st.session_state["cmems_sync_result"] = sync_result
            st.rerun()
    
    with col2:
        if st.button(" 清理所有缓存", use_container_width=True):
            clear_discovery_caches()
            # 清理 Streamlit 缓存
            st.cache_data.clear()
            st.toast("所有缓存已清理")
            st.success(" 缓存已清理")

    if "cmems_sync_result" in st.session_state:
        sync_result = st.session_state.pop("cmems_sync_result")
        if sync_result.get("exit_code") == 0:
            st.success(" CMEMS 同步完成，状态已更新")
            st.toast("CMEMS 同步完成")
        else:
            error_msg = sync_result.get("error", sync_result.get("stderr", "Unknown error"))
            st.error(f" CMEMS 同步失败：{error_msg}")
    
    # 运行数据发现
    with st.spinner("正在扫描 CMEMS 数据..."):
        layers = discover_cmems_layers()
        summary = get_cmems_status_summary(layers)
    
    # 显示摘要
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "已找到",
            f"{summary['found_count']}/{summary['total_count']}",
            delta="正常" if summary['found_count'] == summary['total_count'] else "部分缺失",
            delta_color="normal" if summary['found_count'] == summary['total_count'] else "inverse",
        )
    
    with col2:
        st.metric(
            "缺失",
            summary['missing_count'],
            delta="需要补充" if summary['missing_count'] > 0 else "完整",
            delta_color="inverse" if summary['missing_count'] > 0 else "normal",
        )
    
    with col3:
        if summary['found_count'] == summary['total_count']:
            st.metric("状态", " 完整", delta="所有层已就绪")
        else:
            st.metric("状态", " 不完整", delta=f"缺少 {summary['missing_count']} 层")
    
    # 详细状态表格
    st.markdown("#### 数据层详情")
    
    # 构建表格数据
    table_data = []
    for layer_name, layer_info in layers.items():
        table_data.append({
            "层": layer_name.upper(),
            "状态": " 找到" if layer_info.found else " 缺失",
            "来源": layer_info.source,
            "路径": layer_info.path if layer_info.path else "—",
            "大小": f"{layer_info.size_mb:.1f} MB" if layer_info.size_mb else "—",
            "说明": layer_info.reason,
        })
    
    st.dataframe(table_data, use_container_width=True, hide_index=True)
    
    # 搜索目录说明
    with st.expander(" 搜索目录说明", expanded=False):
        search_dirs = DEFAULT_NEWENV_DIRS + DEFAULT_CACHE_DIRS
        st.markdown(
            f"""
        **??????**
        1. ????????????
        2. `data_processed/newenv/` ?????
        3. `data/cmems_cache/` ????
        4. `data_processed/newenv/cmems_newenv_index.json` ????
        5. `reports/cmems_newenv_index.json` ?????fallback?
        
        **???????**
        {chr(10).join(f'- `{d}`' for d in search_dirs)}
        
        **??????**
        - SIC: `ice_copernicus_sic.nc` ?`sic_latest.nc`
        - SWH: `wave_swh.nc` ?`swh_latest.nc`
        - SIT: `ice_thickness.nc` ?`sit_latest.nc`
        - Drift: `ice_drift.nc` ?`drift_latest.nc`
        
        **Cache ?????**
        - SIC: `*sic*.nc`, `*siconc*.nc`
        - SWH: `*swh*.nc`, `*wave*.nc`
        - SIT: `*thickness*.nc`, `*sit*.nc`
        - Drift: `*drift*.nc`, `*uice*.nc`, `*vice*.nc`
        """
        )
    
    st.markdown("---")
    
    # ========== AIS ???? ==========
    st.subheader("?? AIS ??????")
    st.caption("???????? bathymetry / ports / corridors / AIS ????")

    # ??????
    default_dirs_str = ", ".join(DEFAULT_AIS_DIRS[:3]) + "..."

    ais_dirs_input = st.text_input(
        "??????????",
        value=default_dirs_str,
        help="????????????????????????",
    )

    # ???????
    if ais_dirs_input and ais_dirs_input != default_dirs_str:
        custom_dirs = [d.strip() for d in ais_dirs_input.split(",") if d.strip()]
    else:
        st.error("? ??? AIS ????")
        st.info("**???????**\n" + "\n".join(f"- {d}" for d in search_dirs))
        st.caption("??? AIS ?????.nc ???????????")

    # ??????
    if st.button("?? ???? AIS", use_container_width=True):
        clear_discovery_caches()
        st.cache_data.clear()
        st.toast("?????? AIS ??...")
        st.session_state["ais_scan_requested"] = True
        st.rerun()

    # ?? AIS ??
    with st.spinner("???? AIS ????..."):
        if custom_dirs:
            search_dirs = custom_dirs
        else:
            search_dirs = DEFAULT_AIS_DIRS

        grid_sig = None
        try:
            grid, _, _ = load_grid_with_landmask(prefer_real=True)
            if grid is not None:
                grid_sig = compute_grid_signature(grid)
        except Exception:
            grid_sig = None

        candidates = scan_candidates([Path(d) for d in search_dirs])
        candidates.sort(key=lambda c: c.mtime, reverse=True)
        best, best_meta = select_best_candidate(candidates, grid_sig)

    # ????
    col1, col2 = st.columns(2)

    with col1:
        found_count = len(candidates)
        st.metric(
            "????",
            found_count,
            delta="??" if found_count > 0 else "???",
            delta_color="normal" if found_count > 0 else "inverse",
        )

    with col2:
        if best:
            st.metric(
                "????",
                best.path.name,
                delta=f"??({best.mtime.strftime('%Y-%m-%d')})",
            )
            if grid_sig and not best.signature_matched:
                st.info("?????????????")
        else:
            st.metric("????", "?", delta="???????", delta_color="inverse")

    if st.session_state.pop("ais_scan_requested", False):
        st.success("? AIS ????")
        st.toast("AIS ????")

    # ??????
    if candidates:
        st.markdown("#### ????")

        table_data = []
        for candidate in candidates:
            is_best = (best and candidate.path == best.path)
            table_data.append({
                "??": "?" if is_best else "",
                "???": candidate.path.name,
                "??": str(candidate.path),
                "??": f"{candidate.size_mb:.1f} MB",
                "????": candidate.mtime.strftime("%Y-%m-%d %H:%M:%S"),
                "??": str(candidate.shape) if candidate.shape else "?",
                "signature_matched": "?" if candidate.signature_matched else "",
            })

        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.error("? ??? AIS ????")
        st.info("**???????**\n" + "\n".join(f"- {d}" for d in search_dirs))
        st.caption("??? AIS ?????.nc ???????????")

    # ????
    with st.expander("?? AIS ????", expanded=False):
        st.markdown(f"""
        **???????**
        {chr(10).join(f'- `{d}`' for d in DEFAULT_AIS_DIRS)}

        **?????**
        - ??????`.nc`??
        - ?????????????`density`, `ais`, `traffic`, `corridor`
        - ??????????????

        **???? AIS ???**
        1. ? AIS ?? NetCDF ??????????
        2. ???????`ais`?`density`???
        3. ??"????"??
        """)
    st.markdown("---")
    
    # ========== 静态资产 ==========
    st.subheader(" 静态资产检查")
    
    # Manifest 路径
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
            st.markdown('<span class="status-badge active"> 存在</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge inactive"> 缺失</span>', unsafe_allow_html=True)
    
    st.caption(f"环境变量 ARCTICROUTE_MANIFEST: {manifest_env}")
    
    # 重新扫描按钮
    if st.button(" 运行 Static Assets Doctor", use_container_width=True, type="primary"):
        with st.spinner("正在扫描静态资产..."):
            scan_result = run_static_assets_doctor()
            
            # 保存到 session_state
            st.session_state["static_assets_last_scan"] = scan_result
            
            # 显示结果
            if scan_result["exit_code"] == 0:
                report = scan_result.get("report", {})
                missing_req = len(report.get("missing_required", []))
                missing_opt = len(report.get("missing_optional", []))
                
                st.success(f" 扫描完成：missing_required={missing_req}, missing_optional={missing_opt}")
                st.toast("Static assets doctor: done")
            else:
                error_msg = scan_result.get("error", scan_result.get("stderr", "Unknown error"))
                st.error(f" 扫描失败：exit_code={scan_result['exit_code']}")
                st.error(f"错误信息：{error_msg}")
            
            st.rerun()
    
    # 显示上次扫描结果
    if "static_assets_last_scan" in st.session_state:
        last_scan = st.session_state["static_assets_last_scan"]
        
        st.markdown("#### 上次扫描结果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("退出码", last_scan.get("exit_code", "—"))
        
        with col2:
            st.metric(
                "扫描时间",
                datetime.fromisoformat(last_scan["timestamp"]).strftime("%H:%M:%S")
                if "timestamp" in last_scan else "—"
            )
        
        with col3:
            report = last_scan.get("report", {})
            if report and not report.get("error"):
                missing_req = len(report.get("missing_required", []))
                missing_opt = len(report.get("missing_optional", []))
                st.metric("缺失资产", f"{missing_req + missing_opt}")
            else:
                st.metric("状态", "错误")
        
        # 详细报告
        if "report" in last_scan and not last_scan["report"].get("error"):
            report = last_scan["report"]
            
            if report.get("missing_required"):
                with st.expander(" 缺失的必需资产", expanded=True):
                    for asset in report["missing_required"]:
                        st.text(f"- {asset}")
            
            if report.get("missing_optional"):
                with st.expander(" 缺失的可选资产", expanded=False):
                    for asset in report["missing_optional"]:
                        st.text(f"- {asset}")
    else:
        # 尝试加载现有报告
        doctor_report = load_static_assets_doctor()
        
        if not doctor_report.get("error"):
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
                )
            
            with col3:
                all_ok = doctor_report.get("all_ok", False)
                st.metric(
                    "整体状态",
                    " 正常" if all_ok else " 警告",
                )
        else:
            st.info("点击上方按钮运行 Doctor 检查")
