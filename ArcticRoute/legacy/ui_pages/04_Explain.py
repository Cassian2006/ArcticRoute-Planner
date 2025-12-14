from __future__ import annotations
"""
[LEGACY] Archived multi-page UI file. Not exposed in main UI navigation.
仅供开发者参考/对比。

重新启用方式（不建议）：
1) 将本文件移回 ArcticRoute/pages/04_Explain.py；
2) 在 ArcticRoute/config/runtime.yaml 将 ui.pages.explain 设为 true。
"""
import streamlit as st
from ArcticRoute.apps.pages import explain as _impl  # type: ignore


def render(ctx: dict | None = None) -> None:
    _impl.render(ctx)


if __name__ == "__main__":  # 手动执行时的最小入口
    st.set_page_config(page_title="Explain (Legacy)", layout="wide")
    render()





