from __future__ import annotations
import streamlit as st
from arcticroute.ui.ui_doctor import run_ui_doctor

def render_doctor_page() -> None:
    st.subheader("系统体检")
    st.caption("用于快速定位：问号字符、缺数据、模块不可用、资产缺失等问题。")
    if st.button("重新体检", use_container_width=True):
        st.rerun()

    checks = run_ui_doctor()
    ok = sum(1 for c in checks if c.level == "OK")
    warn = sum(1 for c in checks if c.level == "WARN")
    fail = sum(1 for c in checks if c.level == "FAIL")
    st.write(f"结果：OK={ok}  WARN={warn}  FAIL={fail}")

    for c in checks:
        if c.level == "OK":
            st.success(f"[{c.id}] {c.title} — {c.detail}")
        elif c.level == "WARN":
            with st.expander(f"[WARN {c.id}] {c.title} — {c.detail}", expanded=False):
                st.write("修复建议：")
                st.code(c.fix)
        else:
            with st.expander(f"[FAIL {c.id}] {c.title} — {c.detail}", expanded=True):
                st.write("修复建议：")
                st.code(c.fix)
