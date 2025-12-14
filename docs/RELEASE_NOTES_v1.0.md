# ArcticRoute 1.0 冻结说明

- 冻结日期：2025-11-XX
- Git tag：`v1.0.0`

## 功能范围（简要）

- Planner：A* 栅格规划（已能避陆），可视化带平滑路线。
- 高级风险：fusion_mode / w_interact / use_escort / risk_agg_mode 控件（无数据时自动降级）。
- Eco：可开关的燃油 / CO₂ 评估（eco_model vs 基于距离估算）。
- Review：支持绘制 no-go 区 / 走廊，并重规划第二条路线。
- AI 解释器：根据路线摘要给出中文解释的初版。
- 历史主航线图层：已接线，但位置存在已知偏差问题。

## 已知问题

- 历史主航线显示位置不准确。
- 部分风险数据缺失时会使用兜底成本场，数值更偏“示意”。

## 运行方式（1.0）

```bash
# 从项目根目录：
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-1.0.txt
streamlit run ArcticRoute/ui_app.py
