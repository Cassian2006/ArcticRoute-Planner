from __future__ import annotations
import subprocess
from pathlib import Path
import streamlit as st
from ArcticRoute.apps.registry import UIRegistry  # type: ignore
from ArcticRoute.apps import state as ui_state  # type: ignore
from ArcticRoute.apps.theme import inject_theme, read_theme_flag  # type: ignore
from ArcticRoute.apps.components.page_header import render_page_header  # type: ignore
from ArcticRoute.apps.components.error_card import show_error_card  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_flags() -> tuple[bool, bool]:
    try:
        import yaml  # type: ignore
        yml = (_repo_root() / "ArcticRoute" / "config" / "runtime.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(yml) or {}
        ui_cfg = data.get("ui") or {}
        return bool(ui_cfg.get("task_background", True)), bool(ui_cfg.get("task_persist", True))
    except Exception:
        return True, True


def render(ctx: dict | None = None) -> None:
    inject_theme(read_theme_flag())
    if not UIRegistry().is_page_enabled("reports", False):
        st.info("Reports 页面已在配置中禁用（ui.pages.reports=false）")
        return
    render_page_header("📦", "Reports · 报告构建与下载", "后台任务优先，失败可回退同步执行")
    with st.expander("使用说明 / Usage", expanded=False):
        st.markdown("""
        - 选择 YM（YYYYMM）与包含项 include → 点击“构建报告”。
        - 任务会在下方任务面板显示进度，可下载 HTML/ZIP 结果。
        - CLI 示例：
          - python -m ArcticRoute.api.cli report.build --ym 202412 --include pareto
          - python -m ArcticRoute.api.cli report.build --ym 202412 --include pareto calibration
        """)
    st.markdown('<div class="section"><h2>构建报告</h2><p>选择 YYYYMM，生成统一报告并下载。</p></div>', unsafe_allow_html=True)
    ym = st.text_input("月份 YYYYMM", value=str(st.session_state.get("ym", "202412")))
    with st.expander("说明", expanded=False):
        st.markdown("- 使用 report.build 生成月份报告（pareto/calibration/audit/robust/eco 可按需扩展）。当 ui.task_background=true 时，提交后在后台运行并进入统一任务面板；否则同步执行并提示‘回退模式’。\n- CLI 示例：\n  - python -m ArcticRoute.api.cli report.build --ym 202412 --include pareto\n  - python -m ArcticRoute.api.cli report.build --ym 202412 --include pareto calibration")
    include = st.multiselect("包含项", options=["pareto", "calibration", "audit", "robust", "eco"], default=["pareto"]) 

    bg_on, _persist = _read_flags()

    # 提交后台任务（优先）
    def _submit_background():
        try:
            from ArcticRoute.apps.services import tasks as task_svc  # type: ignore
        except Exception as e:
            st.warning(f"后台任务服务不可用，已回退：{e}")
            return _run_sync()

        def _worker(*, ym_in: str, include_in: list[str], task_ctx: dict):
            import sys
            print(f"[report.build] start ym={ym_in} include={include_in}")
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "report.build", "--ym", ym_in]
            if include_in:
                cmd += ["--include", *include_in]
            import subprocess as sp
            proc = sp.run(cmd, capture_output=True, text=True, cwd=_repo_root())
            if proc.returncode != 0:
                print(proc.stderr)
                raise RuntimeError(f"report.build failed rc={proc.returncode}")
            print(proc.stdout)
            bundle_zip = _repo_root() / "reports" / "bundles" / f"p1_report_{ym_in}.zip"
            return str(bundle_zip) if bundle_zip.exists() else None

        task_id = task_svc.submit_task(
            _worker,
            args=(),
            kwargs={"ym_in": ym, "include_in": include},
            name=f"report.build {ym}",
            kind="report",
            meta={"inputs": {"ym": ym, "include": include}},
        )
        # 维护最近 20 个任务 id
        arr = st.session_state.get("reports_tasks", [])
        arr = [task_id] + [x for x in arr if x != task_id]
        st.session_state["reports_tasks"] = arr[:20]
        st.success(f"已提交后台任务：{task_id}")

    # 同步回退逻辑
    def _run_sync():
        st.info("回退模式：同步执行 report.build（ui.task_background=false 或服务不可用）")
        try:
            import sys
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "report.build", "--ym", ym]
            if include:
                cmd += ["--include", *include]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=_repo_root())
            if res.returncode == 0:
                st.success("report.build 完成")
                try:
                    ui_state.write_action_meta2(
                        action="report.build",
                        inputs={"ym": ym, "include": include},
                        outputs={"report_paths": [], "task_id": ""},
                    )
                except Exception:
                    pass
                st.code(res.stdout[-1600:], language="json")
            else:
                st.error(f"report.build 失败：{res.returncode}")
                st.code(res.stderr[-1600:])
        except Exception as e:
            st.error(str(e))

    if st.button("构建报告", width='stretch'):
        if bg_on:
            _submit_background()
        else:
            _run_sync()

    # 后台任务面板接入 + 轮询
    if bg_on:
        try:
            from ArcticRoute.apps.services import tasks as task_svc  # type: ignore
            st.autorefresh(interval=3000, key="reports-auto-refresh", limit=1_000_000)
            tasks = task_svc.list_tasks(kind="report", limit=100)
            # 写 meta（完成后）与渲染组件
            for t in tasks:
                if t.get("status") == "succeeded":
                    ym_t = (t.get("meta") or {}).get("inputs", {}).get("ym") or ym
                    html_p = _repo_root() / "ArcticRoute" / "reports" / "d_stage" / "phaseG" / f"pareto_{ym_t}_nsr_wbound_smoke.html"
                    zip_p = _repo_root() / "reports" / "bundles" / f"p1_report_{ym_t}.zip"
                    outs = {"report_paths": [str(p) for p in [html_p, zip_p] if p.exists()], "task_id": t.get("id")}
                    try:
                        key = f"_meta_written_{t.get('id')}"
                        if not st.session_state.get(key):
                            ui_state.write_action_meta2(
                                action="report.build",
                                inputs=(t.get("meta") or {}).get("inputs", {}),
                                outputs=outs,
                            )
                            st.session_state[key] = True
                    except Exception:
                        pass
            # 读取日志尾部内容进行展示
            snaps: list[dict] = []
            for s in tasks:
                s2 = dict(s)
                for key in ("stdout", "stderr"):
                    p = s.get(key)
                    if isinstance(p, str) and Path(p).exists():
                        try:
                            content = Path(p).read_text(encoding="utf-8", errors="ignore")[-2000:]
                        except Exception:
                            content = ""
                        s2[key] = content
                snaps.append(s2)
            # 渲染：优先复用组件
            try:
                from ArcticRoute.apps.components.task_panel import render_task_panel as _panel  # type: ignore
                _panel("任务面板（Reports）", snaps, panel_id="reports")
            except Exception:
                st.info("任务面板组件缺失，显示简表（回退）。")
                st.write(snaps)
            # 取消与重试
            with st.expander("管理 · 取消与重试", expanded=False):
                sel = st.selectbox("选择任务", options=[t.get("id") for t in tasks] if tasks else [])
                if sel:
                    t = next((x for x in tasks if x.get("id") == sel), None)
                    c1, c2 = st.columns(2)
                    if t and t.get("status") == "running":
                        if c1.button("取消运行", key=f"cancel-{sel}"):
                            try:
                                task_svc.cancel_task(sel)
                                st.success("已请求取消（软中断）")
                            except Exception as e:
                                st.error(str(e))
                    if t and t.get("status") == "failed":
                        if c2.button("重试此任务", key=f"retry-{sel}"):
                            meta_in = ((t.get("meta") or {}).get("inputs") or {})
                            ym2 = str(meta_in.get("ym", ym))
                            inc2 = list(meta_in.get("include", include))
                            try:
                                def _worker_retry(*, ym_in: str, include_in: list[str], task_ctx: dict):
                                    import sys
                                    print(f"[report.build.retry] ym={ym_in} include={include_in}")
                                    cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "report.build", "--ym", ym_in]
                                    if include_in:
                                        cmd += ["--include", *include_in]
                                    import subprocess as sp
                                    proc = sp.run(cmd, capture_output=True, text=True, cwd=_repo_root())
                                    if proc.returncode != 0:
                                        print(proc.stderr)
                                        raise RuntimeError(f"report.build failed rc={proc.returncode}")
                                    print(proc.stdout)
                                task_svc.submit_task(
                                    _worker_retry,
                                    kwargs={"ym_in": ym2, "include_in": inc2},
                                    name=f"report.build {ym2}",
                                    kind="report",
                                    meta={"inputs": {"ym": ym2, "include": inc2}},
                                )
                                st.success("已重试并提交")
                            except Exception as e:
                                st.error(str(e))
        except Exception as e:
            st.warning(f"后台任务读取失败：{e}")

    # 常用下载位
    pareto_html = _repo_root() / "ArcticRoute" / "reports" / "d_stage" / "phaseG" / f"pareto_{ym}_nsr_wbound_smoke.html"
    bundle_zip = _repo_root() / "reports" / "bundles" / f"p1_report_{ym}.zip"
    c1, c2 = st.columns(2)
    if pareto_html.exists():
        with open(pareto_html, "rb") as fh:
            c1.download_button("下载 Pareto HTML", data=fh.read(), file_name=pareto_html.name, mime="text/html")
    if bundle_zip.exists():
        with open(bundle_zip, "rb") as fh:
            c2.download_button("下载综合 ZIP", data=fh.read(), file_name=bundle_zip.name, mime="application/zip")

    st.markdown("---")
    # 页面底部：冒烟说明链接
    smoke_doc = _repo_root() / "docs" / "smoke" / "SMOKE-UI-MPA.md"
    if smoke_doc.exists():
        st.link_button("打开冒烟说明 (SMOKE-UI-MPA)", url=f"file://{smoke_doc.as_posix()}")
    else:
        st.caption("SMOKE 文档尚未生成。")
