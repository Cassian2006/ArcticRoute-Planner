from __future__ import annotations
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ASSET = Path(__file__).resolve().parents[1] / "assets" / "arctic_ui_cover.html"

_NAV_INJECT = r"""
<script>
(function(){
  function navTo(page){
    try{
      const p = window.top.location.pathname || "/";
      window.top.location.href = p + "?page=" + encodeURIComponent(page);
    }catch(e){
      // fallback: try parent
      try{
        const p = window.parent.location.pathname || "/";
        window.parent.location.href = p + "?page=" + encodeURIComponent(page);
      }catch(e2){}
    }
  }
  function hook(){
    const start = document.getElementById("btnStart");
    if(start){
      start.addEventListener("click", function(ev){
        // 允许原动画先跑一小段再跳转
        setTimeout(()=>navTo("planner"), 420);
      }, {once:false});
    }
    const shot = document.getElementById("btnFakeShot");
    if(shot){
      shot.addEventListener("click", function(ev){
        setTimeout(()=>navTo("data"), 120);
      }, {once:false});
    }
  }
  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", hook);
  }else{
    hook();
  }
})();
</script>
"""

def render_cover_page() -> None:
    st.markdown(
        """
<style>
/* 封面页：尽量铺满，去掉默认边距感 */
.block-container {padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 1200px;}
</style>
""",
        unsafe_allow_html=True,
    )

    if not ASSET.exists():
        st.error(f"Cover asset missing: {ASSET}")
        st.stop()

    html = ASSET.read_text(encoding="utf-8", errors="ignore")
    # 仅追加一次导航脚本
    if "navTo(\"planner\")" not in html:
        html = html + "\n" + _NAV_INJECT

    # height 给一个较大值，避免滚动条/裁切
    components.html(html, height=860, scrolling=False)

