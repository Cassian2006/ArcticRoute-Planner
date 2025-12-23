from __future__ import annotations
from pathlib import Path
import traceback
import datetime as _dt
import streamlit as st

def _write_exception(report_path: Path, page: str, exc: BaseException) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    content = [
        f"=== UI EXCEPTION ===",
        f"time: {ts}",
        f"page: {page}",
        f"type: {type(exc).__name__}",
        f"message: {exc}",
        "",
        traceback.format_exc(),
        ""
    ]
    report_path.write_text("\n".join(content), encoding="utf-8", errors="ignore")

def safe_render(page: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        _write_exception(Path("reports/ui_last_exception.txt"), page, e)
        st.error("页面渲染出现异常：已自动记录到 reports/ui_last_exception.txt（请把该文件内容发回用于定位）。")
        with st.expander("查看简要错误信息"):
            st.code(f"{type(e).__name__}: {e}")
