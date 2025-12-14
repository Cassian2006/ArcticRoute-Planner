# PROJECT LAYOUT

- 主 UI：单页模式（Planner）
  - 入口：`ArcticRoute/ui_app.py`
  - 页面：`ArcticRoute/pages/00_Planner.py`
- 旧版多页面 UI：已收纳到 `legacy`，仅用于对比参考/开发者使用。
  - 存根：`ArcticRoute/legacy/ui_pages/*.py`（带 LEGACY 注释；不出现在主 UI 导航）
  - 原页面实现仍在 `ArcticRoute/apps/pages/*.py` 可被功能复用，避免重复代码。

> 说明：`ArcticRoute/config/runtime.yaml` 中仅保留 `ui.pages.planner: true`，其余页面开关均为 `false`。