"""
UI Shell Skin - 提供统一的 CSS 样式注入
从 arctic_ui_shell.html 提取 CSS 并注入到 Streamlit
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def inject_shell_css() -> None:
    """
    从 arctic_ui_shell.html 提取 CSS 样式并注入到 Streamlit
    
    只提取 CSS，不渲染 HTML 导航（避免双导航问题）
    """
    # 获取 HTML 文件路径
    assets_dir = Path(__file__).parent / "assets"
    shell_html_path = assets_dir / "arctic_ui_shell.html"
    
    if not shell_html_path.exists():
        st.warning(f" UI 壳子文件不存在: {shell_html_path}")
        return
    
    try:
        # 读取 HTML 文件
        html_content = shell_html_path.read_text(encoding="utf-8")
        
        # 提取 <style>...</style> 内容
        import re
        style_match = re.search(r'<style>(.*?)</style>', html_content, re.DOTALL)
        
        if style_match:
            css_content = style_match.group(1)
            
            # 注入 CSS
            st.markdown(
                f"<style>{css_content}</style>",
                unsafe_allow_html=True
            )
        else:
            st.warning(" 未在 arctic_ui_shell.html 中找到 <style> 标签")
            
    except Exception as e:
        st.error(f" 加载 UI 壳子失败: {e}")


def inject_custom_css() -> None:
    """
    注入额外的自定义 CSS（用于特殊需求）
    """
    custom_css = """
    <style>
    /* 额外的自定义样式 */
    
    /* 确保侧边栏滚动条可见 */
    [data-testid="stSidebar"] {
        overflow-y: auto !important;
    }
    
    /* 优化小屏幕显示 */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* 流程管线特殊样式 */
    .pipeline-node {
        background: var(--bg-card, #0b1120);
        border: 1px solid var(--border-color, rgba(255, 255, 255, 0.08));
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .pipeline-node.done {
        border-left: 4px solid var(--accent-green, #10b981);
    }
    
    .pipeline-node.running {
        border-left: 4px solid var(--accent-blue, #38bdf8);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .pipeline-node.fail {
        border-left: 4px solid var(--accent-red, #ef4444);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* 权重面板样式 */
    .weight-panel {
        background: var(--bg-card, #0b1120);
        border: 1px solid var(--border-color, rgba(255, 255, 255, 0.08));
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .weight-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-color, rgba(255, 255, 255, 0.05));
    }
    
    .weight-item:last-child {
        border-bottom: none;
    }
    
    /* 状态徽章 */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-badge.active {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .status-badge.inactive {
        background: rgba(107, 114, 128, 0.2);
        color: #9ca3af;
    }
    
    .status-badge.warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def inject_all_styles() -> None:
    """
    注入所有样式（主壳子 + 自定义）
    """
    inject_shell_css()
    inject_custom_css()

