#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArcticRoute Recon Scanner (只读)
- 扫描项目结构/CLI/前端/环境/安全
- 读取 NetCDF 样例推断网格契约（若 xarray 可用）
- 产出 9 个报告到指定目录
使用：
  python scripts/recon_scan.py --out reports/recon
约束：
- 只读现有文件，不修改任何既有文件
- 遇缺数据/依赖记录告警但不中断
"""
from __future__ import annotations
import os
import re
import sys
import json
import argparse
import fnmatch
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# -------------------- helpers --------------------
IGNORES = {'.git', 'venv', '.venv', '__pycache__', '.mypy_cache', '.ruff_cache', '.pytest_cache', 'node_modules'}
ROOT = Path(__file__).resolve().parents[1]


def _safe_stat(p: Path) -> Dict[str, Any]:
    try:
        st = p.stat()
        return {
            'path': str(p).replace(str(ROOT), '').lstrip(r"/\\"),
            'size': st.st_size,
            'mtime': datetime.fromtimestamp(st.st_mtime).isoformat(),
            'type': 'dir' if stat.S_ISDIR(st.st_mode) else 'file',
        }
    except Exception as e:
        return {'path': str(p), 'error': str(e)}


def _glob_one(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out += list(ROOT.glob(pat))
    # unique & sort by mtime desc
    out = sorted({p.resolve() for p in out if p.exists()}, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return out


# -------------------- required functions --------------------

def _json_default(o):
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore
    # numpy scalar
    if np is not None and isinstance(o, getattr(np, 'generic', ())):
        try:
            return o.item()
        except Exception:
            return str(o)
    # numpy array
    if np is not None and hasattr(o, 'tolist'):
        try:
            return o.tolist()
        except Exception:
            return str(o)
    # datetime
    if isinstance(o, datetime):
        return o.isoformat()
    # bytes
    if isinstance(o, (bytes, bytearray)):
        return o.decode('utf-8', errors='ignore')
    return str(o)

def scan_tree(root: str) -> dict:
    """列出顶层与关键子目录的文件名/大小/mtime；忽略 .git/venv/__pycache__。
    返回：{'root': abs_path, 'entries': [...], 'summary': {...}}
    """
    base = Path(root).resolve()
    result = {
        'root': str(base),
        'entries': [],
        'summary': {
            'has_io': (base / 'ArcticRoute' / 'io').exists() or (base / 'io').exists(),
            'has_core': (base / 'ArcticRoute' / 'core').exists() or (base / 'core').exists(),
            'has_api_cli': (base / 'api' / 'cli.py').exists() or (base / 'ArcticRoute' / 'api' / 'cli.py').exists(),
            'has_apps_app_min': (base / 'ArcticRoute' / 'apps' / 'app_min.py').exists() or (base / 'apps' / 'app_min.py').exists(),
            'has_scripts': (base / 'scripts').exists(),
        }
    }
    # only first two depths
    for p, dirs, files in os.walk(base):
        rel = Path(p).relative_to(base)
        # prune ignored
        dirs[:] = [d for d in dirs if d not in IGNORES]
        if len(rel.parts) > 2:
            continue
        # record directory itself
        result['entries'].append(_safe_stat(Path(p)))
        for f in files:
            if f.startswith('.'):
                continue
            fp = Path(p) / f
            result['entries'].append(_safe_stat(fp))
    return result


def parse_cli_commands(cli_path: str) -> List[dict]:
    """静态解析 Click/Typer 风格命令。
    收集：name, params, help, type(click|typer), has_dry_run
    """
    out: List[dict] = []
    p = Path(cli_path)
    if not p.exists():
        return out
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        text = p.read_text(errors='ignore')

    is_click = 'import click' in text
    is_typer = 'import typer' in text or 'from typer' in text

    # find @click.command / @app.command, and group.command
    cmd_pattern = re.compile(r"@(?:click\.|typer\.)?(?:command|group)\(.*?\)\s*def\s+(\w+)\(", re.S)
    subcmd_pattern = re.compile(r"@\w+\.command\(.*?\)\s*def\s+(\w+)\(", re.S)
    help_pattern = re.compile(r'"""(.*?)"""|\'(.*?)\'', re.S)

    # Fallback: find Typer app and .command decorated functions
    commands = set(m.group(1) for m in cmd_pattern.finditer(text)) | set(m.group(1) for m in subcmd_pattern.finditer(text))

    # Parse parameters for each def name(...):
    for name in sorted(commands):
        # find function signature line
        sig_re = re.compile(rf"def\s+{re.escape(name)}\((.*?)\):", re.S)
        sig_m = sig_re.search(text)
        params = []
        if sig_m:
            raw = sig_m.group(1)
            # crude split respecting simple defaults
            for part in [s.strip() for s in raw.split(',') if s.strip()]:
                params.append(part)
        # try to find nearby help string (docstring right after def)
        help_txt = ''
        # find function block start index
        def_idx = text.find(f"def {name}(")
        if def_idx != -1:
            doc_m = re.search(r"def\s+%s\([\s\S]*?\):\s*\n\s*(?:\"\"\"([\s\S]*?)\"\"\"|\'\'\'([\s\S]*?)\'\'\')" % re.escape(name), text)
            if doc_m:
                help_txt = (doc_m.group(1) or doc_m.group(2) or '').strip()
        has_dry = '--dry-run' in text or 'dry_run' in (','.join(params))
        out.append({
            'name': name,
            'params': params,
            'help': help_txt,
            'framework': 'click' if is_click else ('typer' if is_typer else 'unknown'),
            'has_dry_run': has_dry,
        })
    return out


def detect_grid_spec(sample_glob: str = '') -> dict:
    """从样例 NetCDF 推断 dims/coords/attrs，时间频率与空间分辨率。
    优先匹配：data_processed/ice_forecast/merged/*.nc 或 data_processed/**/ice_cost_*.nc
    若 xarray 不可用或无文件，返回 available: false
    """
    patterns = [
        'data_processed/ice_forecast/merged/*.nc',
        'data_processed/**/ice_cost_*.nc',
        'ArcticRoute/data_processed/ice_forecast/merged/*.nc',
        'ArcticRoute/data_processed/**/ice_cost_*.nc',
        'data_processed/*.nc',
        'ArcticRoute/data_processed/*.nc',
    ]
    files = _glob_one(patterns)
    if not files:
        return {'available': False, 'reason': 'no_netcdf'}
    sample = files[0]
    try:
        import xarray as xr  # type: ignore
    except Exception as e:
        return {'available': False, 'reason': f'xarray_not_available: {e}', 'sample': str(sample)}
    info: Dict[str, Any] = {'available': True, 'sample': str(sample)}
    try:
        ds = xr.open_dataset(sample, engine=None)
        # use sizes and cast to built-in int
        info['dims'] = {k: int(v) for k, v in ds.sizes.items()}
        # coords basic
        coords = {}
        for c in list(ds.coords):
            try:
                v = ds[c]
                coords[c] = {'dims': tuple(v.dims), 'size': int(v.size)}
            except Exception:
                pass
        info['coords'] = coords
        # attrs
        info['attrs'] = dict(ds.attrs)
        # time frequency
        freq = None
        if 'time' in ds.coords:
            try:
                t = ds['time'].to_index()
                if len(t) >= 3:
                    dt = (t[2] - t[1]).astype('timedelta64[h]').astype(int)
                    if dt % 24 == 0:
                        freq = f"{dt//24}D"
                    else:
                        freq = f"{dt}H"
            except Exception:
                pass
        info['freq'] = freq
        # spatial resolution (approx from lat/lon spacing)
        res = None
        latname = 'lat' if 'lat' in ds.variables else ('latitude' if 'latitude' in ds.variables else None)
        lonname = 'lon' if 'lon' in ds.variables else ('longitude' if 'longitude' in ds.variables else None)
        if latname and lonname:
            try:
                latv = ds[latname]
                if latv.ndim == 1 and latv.size >= 2:
                    d = float(abs(latv[1] - latv[0]))
                    res = f"{round(d, 4)}°"
                elif latv.ndim == 2 and latv.shape[0] >= 2 and latv.shape[1] >= 2:
                    d = float(abs(latv[0,1] - latv[0,0]))
                    res = f"~{round(d, 4)}°"
            except Exception:
                pass
        info['resolution'] = res
        ds.close()
    except Exception as e:
        return {'available': False, 'reason': f'read_error: {e}', 'sample': str(sample)}
    return info


def list_p1_artifacts() -> dict:
    """探测预测/成本层/报告/缓存索引的样例路径与计数。"""
    exists = {}
    patterns = {
        'sic_forecast': [
            'data_processed/ice_forecast/merged/*.nc',
            'ArcticRoute/data_processed/ice_forecast/merged/*.nc',
            'data_processed/sic_fcst_*.nc',
        ],
        'ice_cost': [
            'data_processed/**/ice_cost_*.nc',
            'ArcticRoute/data_processed/**/ice_cost_*.nc',
        ],
        'reports_html': [
            'reports/*.html', 'ArcticRoute/reports/*.html'
        ],
        'cache_index': [
            'cache_index.json', 'ArcticRoute/cache_index.json', 'outputs/pipeline_runs.log'
        ],
    }
    for k, pats in patterns.items():
        files = _glob_one(pats)
        exists[k] = {
            'count': len(files),
            'samples': [str(p) for p in files[:5]],
        }
    return exists


def map_streamlit_ui(app_path: str) -> dict:
    """静态解析 Streamlit 应用，提取控件/图层/导出按钮。"""
    p = Path(app_path)
    if not p.exists():
        return {'available': False, 'reason': 'missing_app'}
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        text = p.read_text(errors='ignore')
    ui = {
        'available': True,
        'widgets': [],
        'layers': [],
        'exports': [],
    }
    # widgets
    widget_calls = ['slider', 'selectbox', 'multiselect', 'checkbox', 'radio', 'text_input', 'number_input', 'date_input', 'button']
    for w in widget_calls:
        for m in re.finditer(rf"st\.{w}\((.*?)\)", text):
            ui['widgets'].append({'type': w, 'args': (m.group(1) or '').strip()[:200]})
    # download/export
    for m in re.finditer(r"st\.download_button\((.*?)\)", text):
        ui['exports'].append({'type': 'download_button', 'args': (m.group(1) or '').strip()[:200]})
    for m in re.finditer(r"export|save|to_(?:csv|json|html|png|geojson)\(", text):
        ui['exports'].append({'type': 'export_call', 'match': m.group(0)})
    # layers heuristics: folium/pydeck/leafmap
    if 'st.map' in text or 'pydeck' in text or 'st.pydeck_chart' in text:
        ui['layers'].append('pydeck/map')
    if 'folium' in text or 'leafmap' in text:
        ui['layers'].append('folium/leafmap')
    # custom layer keywords
    for kw in ['route', 'risk', 'ice', 'cost', 'corridor']:
        if re.search(kw, text, re.I):
            ui['layers'].append(f'kw:{kw}')
    # dedup
    ui['widgets'] = ui['widgets'][:50]
    ui['exports'] = ui['exports'][:50]
    ui['layers'] = sorted(set(ui['layers']))
    return ui


def collect_env_info() -> dict:
    """记录 Python/OS、包版本、GPU信息、git 分支/commit。"""
    info: Dict[str, Any] = {}
    info['python'] = sys.version.split()[0]
    import platform
    info['os'] = platform.platform()
    # packages
    versions = {}
    try:
        from importlib.metadata import version, PackageNotFoundError  # py3.8+
    except Exception:
        from importlib_metadata import version, PackageNotFoundError  # type: ignore
    for pkg in ['xarray', 'numpy', 'torch', 'statsmodels', 'streamlit', 'matplotlib']:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
        except Exception as e:
            versions[pkg] = f'err:{e}'
    info['versions'] = versions
    # torch cuda
    torch_info = {
        'installed': versions.get('torch') is not None,
        'cuda_available': False,
        'device_count': 0,
        'capabilities': [],
    }
    if versions.get('torch'):
        try:
            import torch  # type: ignore
            torch_info['cuda_available'] = bool(torch.cuda.is_available())
            torch_info['device_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            caps = []
            for i in range(torch_info['device_count']):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                caps.append({'index': i, 'name': name, 'capability': cap})
            torch_info['capabilities'] = caps
        except Exception as e:
            torch_info['error'] = str(e)
    info['torch'] = torch_info
    # git
    git = {'enabled': (ROOT / '.git').exists(), 'branch': None, 'commit': None}
    if git['enabled']:
        try:
            b = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=str(ROOT)).decode().strip()
            s = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(ROOT)).decode().strip()
            git['branch'] = b
            git['commit'] = s
        except Exception as e:
            git['error'] = str(e)
    info['git'] = git
    return info


def check_security() -> dict:
    """存在性/关键关键词检测：.env.template、SECURITY.md、pre-commit 配置、日志格式关键词（mask/redact）。"""
    checks: Dict[str, Any] = {}
    paths = {
        'env_template': ['.env.template', 'ArcticRoute/.env.template'],
        'security_md': ['SECURITY.md', 'ArcticRoute/SECURITY.md'],
        'pre_commit': ['.pre-commit-config.yaml', '.pre-commit-config.yml', 'pyproject.toml'],
        'logging_cfg': ['logging_config.py', 'ArcticRoute/logging_config.py']
    }
    for k, pats in paths.items():
        files = _glob_one(pats)
        checks[k] = {'exists': bool(files), 'paths': [str(p) for p in files]}
    # keyword scan for logging redact/mask
    redact_keywords = ['redact', 'mask', 'filter', 'PII', 'sanit']
    redact_found = []
    for f in _glob_one(paths['logging_cfg']):
        try:
            txt = Path(f).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            txt = Path(f).read_text(errors='ignore')
        for kw in redact_keywords:
            if kw.lower() in txt.lower():
                redact_found.append({'file': str(f), 'keyword': kw})
    checks['logging_redact_keywords'] = redact_found
    return checks


def summarize_gaps(structure: dict, cli_map: List[dict], grid_spec: dict, p1_artifacts: dict, ui_map: dict, env: dict, sec: dict) -> dict:
    """基于结果生成缺失项与建议（结构化）。"""
    gaps: Dict[str, Any] = {'warnings': [], 'risks': [], 'suggestions': []}
    # structure
    s = structure.get('summary', {})
    if not s.get('has_io'): gaps['warnings'].append('缺少 io/ 目录或模块（可能位于 ArcticRoute/io 或根 io）')
    if not s.get('has_core'): gaps['warnings'].append('缺少 core/ 目录（predictors/cost/route 等）或路径不标准')
    if not s.get('has_api_cli'): gaps['warnings'].append('缺少 api/cli.py（CLI 入口）')
    if not s.get('has_apps_app_min'): gaps['warnings'].append('缺少 apps/app_min.py（Streamlit 前端）')
    if not s.get('has_scripts'): gaps['warnings'].append('缺少 scripts/ 目录')

    # grid
    if not grid_spec.get('available'):
        gaps['risks'].append(f"网格契约不可解析：{grid_spec.get('reason')}")
        gaps['suggestions'].append('提供一个样例 NetCDF (merged/*.nc 或 ice_cost_*.nc)，并确保可用 xarray/netCDF4')
    else:
        dims = grid_spec.get('dims', {})
        if not ({'time'} <= set(dims.keys())):
            gaps['warnings'].append('NetCDF 未包含 time 维度或未识别')
        if grid_spec.get('freq') is None:
            gaps['warnings'].append('时间步长未能推断（样本时间点不足或未标准化）')
        if grid_spec.get('resolution') is None:
            gaps['warnings'].append('空间分辨率未能推断（lat/lon 维度/坐标不规则）')

    # P1 artifacts
    if p1_artifacts.get('sic_forecast', {}).get('count', 0) == 0:
        gaps['risks'].append('缺少海冰预测产物（sic_fcst 或 merged/）样例')
    if p1_artifacts.get('ice_cost', {}).get('count', 0) == 0:
        gaps['risks'].append('缺少冰险成本层产物（ice_cost_*.nc）样例')
    if p1_artifacts.get('reports_html', {}).get('count', 0) == 0:
        gaps['warnings'].append('缺少 HTML 报告样例（run_report_*.html）')
    if p1_artifacts.get('cache_index', {}).get('count', 0) == 0:
        gaps['warnings'].append('缺少缓存索引（cache_index.json 或 pipeline_runs.log）')

    # env
    if env.get('versions', {}).get('xarray') is None:
        gaps['warnings'].append('未安装 xarray，无法解析 NetCDF')
    if env.get('versions', {}).get('torch') is None:
        gaps['warnings'].append('未安装 torch（如需 GPU 推理可忽略）')

    # security
    if not sec.get('env_template', {}).get('exists'):
        gaps['warnings'].append('缺少 .env.template（建议提供）')
    if not sec.get('security_md', {}).get('exists'):
        gaps['warnings'].append('缺少 SECURITY.md（建议补充安全策略）')
    if not sec.get('pre_commit', {}).get('exists'):
        gaps['warnings'].append('缺少 pre-commit 配置（建议引入基础钩子：ruff/black/secretlint）')
    if not sec.get('logging_redact_keywords'):
        gaps['warnings'].append('日志未检测到脱敏关键词（redact/mask），请确认日志中无敏感信息泄露')

    # UI
    if not ui_map.get('available'):
        gaps['warnings'].append('未找到 Streamlit app_min.py 或无法解析')
    elif not ui_map.get('widgets'):
        gaps['warnings'].append('前端控件较少或无法检测到（请检查 st.slider/st.button 等）')

    return gaps


def emit_reports(out_dir: str, payloads: dict) -> None:
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # filenames mapping
    files = {
        'structure': 'structure.json',
        'cli_map': 'cli_map.json',
        'grid_spec': 'grid_spec.json',
        'p1_artifacts': 'p1_artifacts.json',
        'ui_map': 'ui_map.json',
        'env': 'env.json',
        'security': 'security_gaps.md',  # markdown
        'gaps': 'gaps.json',
        'status_md': 'RECON_STATUS.md',
    }

    # write json blobs
    def dump_json(name: str, data: Any):
        (outdir / files[name]).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding='utf-8')

    dump_json('structure', payloads['structure'])
    dump_json('cli_map', payloads['cli_map'])
    dump_json('grid_spec', payloads['grid_spec'])
    dump_json('p1_artifacts', payloads['p1_artifacts'])
    dump_json('ui_map', payloads['ui_map'])
    dump_json('env', payloads['env'])
    dump_json('gaps', payloads['gaps'])

    # security markdown
    sec = payloads['security']
    sec_lines = [
        '# Security/Compliance Checks',
        '',
        f"- .env.template: {'YES' if sec.get('env_template',{}).get('exists') else 'NO'} {sec.get('env_template',{}).get('paths','')}",
        f"- SECURITY.md: {'YES' if sec.get('security_md',{}).get('exists') else 'NO'} {sec.get('security_md',{}).get('paths','')}",
        f"- pre-commit: {'YES' if sec.get('pre_commit',{}).get('exists') else 'NO'} {sec.get('pre_commit',{}).get('paths','')}",
        f"- logging redact keywords: {len(sec.get('logging_redact_keywords', []))} hits",
    ]
    (outdir / files['security']).write_text("\n".join(sec_lines), encoding='utf-8')

    # status markdown
    env = payloads['env']
    grid = payloads['grid_spec']
    gaps = payloads['gaps']
    structure = payloads['structure']
    cli_map = payloads['cli_map']
    ui_map = payloads['ui_map']
    p1 = payloads['p1_artifacts']

    py = env.get('python')
    osver = env.get('os')
    torch = env.get('torch', {})
    gpu_flag = torch.get('cuda_available', False)
    gpu_desc = ''
    if torch.get('capabilities'):
        dev0 = torch['capabilities'][0]
        gpu_desc = f"{dev0.get('name')} {dev0.get('capability')}"
    ver = env.get('versions', {})
    git = env.get('git', {})
    branch = git.get('branch') or 'N/A'
    commit = git.get('commit') or 'N/A'
    date = datetime.now().isoformat(timespec='seconds')

    # P0/P1 进度推断（简单规则）
    p1_state = '缺失'
    if p1.get('sic_forecast', {}).get('count', 0) > 0 and p1.get('ice_cost', {}).get('count', 0) > 0:
        p1_state = '完成'
    elif any(p1.get(k, {}).get('count', 0) > 0 for k in ['sic_forecast', 'ice_cost', 'reports_html', 'cache_index']):
        p1_state = '部分'

    header = [
        '# ArcticRoute Recon Status (只读体检)',
        '',
        f"- 项目根: {structure.get('root')}",
        f"- 分支/提交: {branch} @ {commit} ({date})",
        f"- Python/OS/GPU: {py} | {osver} | GPU={gpu_flag} {gpu_desc}",
        f"- 版本栈: xarray={ver.get('xarray')} , numpy={ver.get('numpy')} , torch={ver.get('torch')} , statsmodels={ver.get('statsmodels')} , streamlit={ver.get('streamlit')} , matplotlib={ver.get('matplotlib')}",
        f"- P0/P1 进度推断: {p1_state}（依据: p1_artifacts）",
        f"- 网格/时间轴: dims={grid.get('dims')} , coords={list(grid.get('coords', {}).keys()) if grid.get('coords') else None} , freq={grid.get('freq')} , resolution={grid.get('resolution')}",
        '',
    ]

    # Sections
    sections = []
    # 结构地图
    entries = payloads['structure'].get('entries', [])
    entries_preview = '\n'.join(f"- {e.get('type')} {e.get('path')} ({e.get('size','?')} bytes, {e.get('mtime','?')})" for e in entries[:80])
    sections.append("## 结构地图\n" + entries_preview)

    # CLI 列表
    if not cli_map:
        sections.append("## CLI 列表\n未检测到命令（api/cli.py）")
    else:
        cli_lines = ["## CLI 列表"]
        for c in cli_map:
            cli_lines.append(f"- {c['name']}({', '.join(c['params'])}) framework={c['framework']} dry_run={c['has_dry_run']}")
            if c.get('help'):
                help_snip = (c.get('help') or '')[:200].replace('\n', ' ')
                cli_lines.append(f"  - {help_snip}")
        sections.append("\n".join(cli_lines))

    # P1 产物
    p1_lines = ["## P1 产物"]
    for k, v in p1.items():
        p1_lines.append(f"- {k}: count={v.get('count')} samples={v.get('samples', [])}")
    sections.append("\n".join(p1_lines))

    # 前端要素
    ui_lines = ["## 前端要素 (Streamlit)"]
    ui_lines.append(f"- widgets: {ui_map.get('widgets')}")
    ui_lines.append(f"- layers: {ui_map.get('layers')}")
    ui_lines.append(f"- exports: {ui_map.get('exports')}")
    sections.append("\n".join(ui_lines))

    # 安全/合规
    sec_lines2 = ["## 安全与合规", f"- 详见: {files['security']}"]
    sections.append("\n".join(sec_lines2))

    # 缺口 & 风险
    gap_lines = ["## 缺口 & 风险", json.dumps(gaps, indent=2, ensure_ascii=False)]
    sections.append("\n".join(gap_lines))

    # 回答关键问题
    q_lines = [
        "## 需要回答的关键问题",
        f"- 项目结构：io/, core/, api/cli.py, apps/app_min.py, scripts/ 是否存在？ -> {structure.get('summary')}",
        f"- P0/P1 进度：P1 的预测/代价层/路由/报告/缓存是否可用？ -> {p1_state}，详见 p1_artifacts",
        f"- 网格/时间轴契约：dims/coords/freq/resolution -> dims={grid.get('dims')} coords={list(grid.get('coords', {}).keys()) if grid.get('coords') else None} freq={grid.get('freq')} res={grid.get('resolution')}",
        f"- 可复用资产：对齐工具/A* 路由/成本映射/报告缓存 -> 依据代码结构与 tests/（需要人工复核）",
        f"- CLI 入口：子命令/参数签名/--dry-run -> 见 CLI 列表",
        f"- Streamlit 前端：图层/控件/导出 -> 见 前端要素",
        f"- 安全与合规：.env.template/SECURITY.md/日志脱敏/路径守卫 -> 见 安全与合规",
        f"- 数据现状：data_processed/ 下是否存在样例 -> 见 P1 产物",
        f"- 对齐函数：ensure_common_grid/align_time 是否存在 -> 需在代码中进一步检索（本脚本未做深度语义扫描）",
        f"- 风险与空白：缺哪些会阻碍 Phase B/C -> 见 缺口 & 风险",
        '',
        '请将本文件全文粘贴到聊天，或提供其路径：reports/recon/RECON_STATUS.md。'
    ]

    status_md = "\n".join(header + sections + q_lines)
    (outdir / files['status_md']).write_text(status_md, encoding='utf-8')


# -------------------- CLI --------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='ArcticRoute 项目只读体检')
    parser.add_argument('--out', default='reports/recon', help='输出目录（默认 reports/recon）')
    args = parser.parse_args(argv)

    # structure
    structure = scan_tree(str(ROOT))

    # cli map: 优先根 api/cli.py，其次 ArcticRoute/api/cli.py
    cli_files = [ROOT / 'api' / 'cli.py', ROOT / 'ArcticRoute' / 'api' / 'cli.py']
    cli_map: List[dict] = []
    for f in cli_files:
        cli_map += parse_cli_commands(str(f))

    grid_spec = detect_grid_spec()
    p1_artifacts = list_p1_artifacts()
    # app path
    app_path = (ROOT / 'apps' / 'app_min.py') if (ROOT / 'apps' / 'app_min.py').exists() else (ROOT / 'ArcticRoute' / 'apps' / 'app_min.py')
    ui_map = map_streamlit_ui(str(app_path)) if app_path.exists() else {'available': False, 'reason': 'not_found'}
    env = collect_env_info()
    sec = check_security()
    gaps = summarize_gaps(structure, cli_map, grid_spec, p1_artifacts, ui_map, env, sec)

    payloads = {
        'structure': structure,
        'cli_map': cli_map,
        'grid_spec': grid_spec,
        'p1_artifacts': p1_artifacts,
        'ui_map': ui_map,
        'env': env,
        'security': sec,
        'gaps': gaps,
    }

    emit_reports(args.out, payloads)
    print(f"Recon reports written to: {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

