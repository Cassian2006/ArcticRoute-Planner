from __future__ import annotations

from pathlib import Path
import base64

import streamlit as st
import streamlit.components.v1 as components


def render() -> None:
    """
    封面页：精简展示，只保留标题 / 标语与核心要点 chips。
    移除原有“工程可回退/可解释/可复现实验”等长文案以及开始任务按钮。
    Logo 使用桌面 roundLOGO.png（如缺失则回退为文字块）。
    """
    logo_path = Path(r"C:\Users\sgddsf\Desktop\roundLOGO.png")
    logo_data = ""
    if logo_path.exists():
        try:
            logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
            logo_data = f"data:image/png;base64,{logo_b64}"
        except Exception:
            logo_data = ""

    html = """
<style>
  :root {{
    --bg:#050910;
    --card:#0c162b;
    --chip:#0b213f;
    --accent:#38bdf8;
    --text:#e2e8f0;
    --muted:#9fb3c8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    min-height: 100vh;
    color: var(--text);
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    padding: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0c1323 0%, #0f1c33 100%);
  }}
  .container {{
    position: relative;
    width: 100%;
    min-height: 90vh;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    isolation: isolate;
  }}
  .container::after {{
    content: "";
    position: absolute;
    inset: 0;
    z-index: 0;
    background-image: radial-gradient(
      ellipse 1.5px 2px at 1.5px 50%,
      #0000 0,
      #0000 90%,
      #000 100%
    );
    background-size: 25px 8px;
    opacity: 0.02;
    pointer-events: none;
  }}
  .container {{
    --c: #9bd5ff;
    background-color: transparent;
    background-image: radial-gradient(4px 100px at 0px 235px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 235px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 117.5px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 252px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 252px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 126px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 150px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 150px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 75px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 253px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 253px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 126.5px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 204px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 204px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 102px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 134px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 134px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 67px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 179px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 179px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 89.5px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 299px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 299px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 149.5px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 215px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 215px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 107.5px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 281px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 281px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 140.5px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 158px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 158px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 79px, var(--c) 100%, #0000 150%),
      radial-gradient(4px 100px at 0px 210px, var(--c), #0000),
      radial-gradient(4px 100px at 300px 210px, var(--c), #0000),
      radial-gradient(1.5px 1.5px at 150px 105px, var(--c) 100%, #0000 150%);
    background-size:
      300px 235px, 300px 235px, 300px 235px,
      300px 252px, 300px 252px, 300px 252px,
      300px 150px, 300px 150px, 300px 150px,
      300px 253px, 300px 253px, 300px 253px,
      300px 204px, 300px 204px, 300px 204px,
      300px 134px, 300px 134px, 300px 134px,
      300px 179px, 300px 179px, 300px 179px,
      300px 299px, 300px 299px, 300px 299px,
      300px 215px, 300px 215px, 300px 215px,
      300px 281px, 300px 281px, 300px 281px,
      300px 158px, 300px 158px, 300px 158px,
      300px 210px, 300px 210px, 300px 210px;
    animation: hi 150s linear infinite;
    z-index: 0;
    pointer-events: none;
  }
  @keyframes hi {{
    0% {{
      background-position:
        0px 220px, 3px 220px, 151.5px 337.5px,
        25px 24px, 28px 24px, 176.5px 150px,
        50px 16px, 53px 16px, 201.5px 91px,
        75px 224px, 78px 224px, 226.5px 350.5px,
        100px 19px, 103px 19px, 251.5px 121px,
        125px 120px, 128px 120px, 276.5px 187px,
        150px 31px, 153px 31px, 301.5px 120.5px,
        175px 235px, 178px 235px, 326.5px 384.5px,
        200px 121px, 203px 121px, 351.5px 228.5px,
        225px 224px, 228px 224px, 376.5px 364.5px,
        250px 26px, 253px 26px, 401.5px 105px,
        275px 75px, 278px 75px, 426.5px 180px;
    }}
    100% {{
      background-position:
        0px 6800px, 3px 6800px, 151.5px 6917.5px,
        25px 13632px, 28px 13632px, 176.5px 13758px,
        50px 5416px, 53px 5416px, 201.5px 5491px,
        75px 17175px, 78px 17175px, 226.5px 17301.5px,
        100px 5119px, 103px 5119px, 251.5px 5221px,
        125px 8428px, 128px 8428px, 276.5px 8495px,
        150px 9876px, 153px 9876px, 301.5px 9965.5px,
        175px 13391px, 178px 13391px, 326.5px 13540.5px,
        200px 14741px, 203px 14741px, 351.5px 14848.5px,
        225px 18770px, 228px 18770px, 376.5px 18910.5px,
        250px 5082px, 253px 5082px, 401.5px 5161px,
        275px 6375px, 278px 6375px, 426.5px 6480px;
    }}
  }}
  .content {{
    position: relative;
    z-index: 2;
    width: 100%;
    max-width: 1040px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 24px;
    text-align: center;
  }}
  .logo-big {{
    width: 180px;
    height: 180px;
    border-radius: 28px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.35);
    display: grid;
    place-items: center;
    overflow: hidden;
    color: #38bdf8;
    font-weight: 900;
    font-size: 28px;
    box-shadow: 0 10px 36px rgba(0,0,0,0.25);
    backdrop-filter: blur(4px);
  }}
  .logo-big img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
  }}
  .glass-card {{
    width: 100%;
    max-width: 960px;
    background: linear-gradient(180deg, rgba(255,255,255,0.36), rgba(215,240,235,0.28));
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 24px;
    padding: 28px 32px;
    box-shadow: 0 16px 42px rgba(0,0,0,0.14), 0 0 0 1px rgba(56,189,248,0.14);
    backdrop-filter: blur(12px);
    color: #0a0f1a;
  }}
  .glass-card h1 {{
    margin: 0 0 12px 0;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 0.3px;
    color: #0a0f1a;
  }}
  .chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
  }}
  .chip {{
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(56,189,248,0.26);
    color: #0b1220;
    padding: 8px 12px;
    border-radius: 12px;
    font-weight: 700;
    letter-spacing: 0.1px;
    font-size: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  }}
  @media (max-width: 640px) {{
    .glass-card {{
      padding: 24px;
    }}
    .logo-big {{
      width: 220px;
      height: 220px;
    }}
  }}
</style>
<div class="container">
  <div class="content">
    <div class="logo-big">__LOGO__</div>
    <div class="glass-card">
      <h1>ArcticRoute</h1>
      <div class="chips">
        <span class="chip">CMEMS 近实时</span>
        <span class="chip">POLARIS 约束</span>
        <span class="chip">多目标 Pareto</span>
        <span class="chip">PolarRoute 可选</span>
      </div>
    </div>
  </div>
</div>
"""

    # 将双花括号还原为单花括号，避免 CSS 失效
    html = html.replace("{{", "{").replace("}}", "}")

    logo_html = (
        f"<img src='{logo_data}' alt='logo' />" if logo_data else "AR"
    )
    components.html(html.replace("__LOGO__", logo_html), height=480, scrolling=False)
