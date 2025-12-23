# -*- coding: utf-8 -*-
"""
评估结果展示模块 (Phase EVAL-UI)

展示 eval_scenario_results 生成的评估结果，包括：
1. 每个 scenario 下 efficient / edl_safe / edl_robust 的 Δ距离、Δ成本、风险下降百分比
2. 散点图：距离增加% vs 风险下降%
3. 自动生成的总体结论文字
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


def load_eval_results() -> Optional[pd.DataFrame]:
    """
    加载评估结果 CSV 文件。
    
    Returns:
        DataFrame 或 None（如果文件不存在）
    """
    eval_path = Path(__file__).resolve().parents[2] / "reports" / "eval_mode_comparison.csv"
    
    if not eval_path.exists():
        return None
    
    try:
        df = pd.read_csv(eval_path)
        return df
    except Exception as e:
        st.warning(f"加载评估结果失败：{e}")
        return None


def generate_global_summary(df: pd.DataFrame) -> dict:
    """
    根据评估数据生成全局总结。
    
    Args:
        df: 评估结果 DataFrame
    
    Returns:
        包含各模式统计信息的字典
    """
    summary = {}
    
    for mode in ["edl_safe", "edl_robust"]:
        mode_df = df[df["mode"] == mode]
        
        if mode_df.empty:
            continue
        
        # 计算有效的风险下降数据（排除 NaN）
        risk_valid = mode_df[pd.notna(mode_df["risk_reduction_pct"])]
        
        if len(risk_valid) > 0:
            avg_risk_red = risk_valid["risk_reduction_pct"].mean()
            avg_rel_dist = mode_df["rel_dist_pct"].mean()
            
            count_better_risk = (risk_valid["risk_reduction_pct"] > 0).sum()
            count_better_risk_small_detour = (
                (risk_valid["risk_reduction_pct"] > 0)
                & (mode_df["rel_dist_pct"] <= 5.0)
            ).sum()
            
            summary[mode] = {
                "avg_risk_reduction": avg_risk_red,
                "avg_distance_increase": avg_rel_dist,
                "scenarios_with_better_risk": count_better_risk,
                "better_risk_small_detour": count_better_risk_small_detour,
                "total_scenarios": len(mode_df),
            }
    
    return summary


def generate_conclusion_text(summary: dict) -> str:
    """
    根据全局总结生成结论文字。
    
    Args:
        summary: 全局总结字典
    
    Returns:
        结论文字
    """
    if not summary:
        return "暂无评估数据。"
    
    lines = []
    lines.append("## 📊 评估结论\n")
    
    # EDL-Safe 结论
    if "edl_safe" in summary:
        safe_stats = summary["edl_safe"]
        lines.append("### EDL-Safe 方案")
        lines.append(
            f"- **平均风险下降**: {safe_stats['avg_risk_reduction']:.1f}%"
        )
        lines.append(
            f"- **平均距离增加**: {safe_stats['avg_distance_increase']:.2f}%"
        )
        lines.append(
            f"- **改善场景数**: {safe_stats['scenarios_with_better_risk']}/{safe_stats['total_scenarios']}"
        )
        lines.append(
            f"- **小绕路改善**: {safe_stats['better_risk_small_detour']}/{safe_stats['total_scenarios']} "
            f"（距离增加 ≤ 5%）"
        )
        
        # 生成定性评价
        if safe_stats['avg_risk_reduction'] > 50:
            lines.append(
                "\n💡 **评价**: EDL-Safe 方案在风险下降和距离增加之间取得良好平衡，"
                "适合追求安全性与经济性兼顾的用户。"
            )
        else:
            lines.append(
                "\n💡 **评价**: EDL-Safe 方案提供了适度的风险下降，"
                "但距离增加较为明显，适合风险敏感型用户。"
            )
        
        lines.append("")
    
    # EDL-Robust 结论
    if "edl_robust" in summary:
        robust_stats = summary["edl_robust"]
        lines.append("### EDL-Robust 方案")
        lines.append(
            f"- **平均风险下降**: {robust_stats['avg_risk_reduction']:.1f}%"
        )
        lines.append(
            f"- **平均距离增加**: {robust_stats['avg_distance_increase']:.2f}%"
        )
        lines.append(
            f"- **改善场景数**: {robust_stats['scenarios_with_better_risk']}/{robust_stats['total_scenarios']}"
        )
        lines.append(
            f"- **小绕路改善**: {robust_stats['better_risk_small_detour']}/{robust_stats['total_scenarios']} "
            f"（距离增加 ≤ 5%）"
        )
        
        # 生成定性评价
        if robust_stats['avg_risk_reduction'] > 75:
            lines.append(
                "\n💡 **评价**: EDL-Robust 方案提供了显著的风险下降，"
                "虽然距离增加较多，但对于高风险厌恶型用户或关键航线非常有价值。"
            )
        else:
            lines.append(
                "\n💡 **评价**: EDL-Robust 方案在风险下降上表现中等，"
                "距离增加代价较大，建议结合具体场景选择使用。"
            )
        
        lines.append("")
    
    # 整体建议
    if "edl_safe" in summary and "edl_robust" in summary:
        safe_stats = summary["edl_safe"]
        robust_stats = summary["edl_robust"]
        
        lines.append("### 🎯 整体建议")
        
        if safe_stats['avg_risk_reduction'] > 50 and safe_stats['avg_distance_increase'] < 5:
            lines.append(
                "- **推荐 EDL-Safe**: 在大多数场景下提供了良好的风险-成本权衡。"
            )
        
        if robust_stats['avg_risk_reduction'] > 75:
            lines.append(
                "- **EDL-Robust 用于高风险场景**: 当需要最大化安全性时使用。"
            )
        
        lines.append(
            "- **场景化选择**: 不同航线的风险特征差异大，建议根据具体场景选择合适的模式。"
        )
    
    return "\n".join(lines)


def render_scenario_table(df: pd.DataFrame) -> None:
    """
    渲染场景对比表格。
    
    Args:
        df: 评估结果 DataFrame
    """
    st.subheader("📋 场景对比详情")
    
    # 按 scenario_id 分组显示
    scenarios = sorted(df["scenario_id"].unique())
    
    for scenario_id in scenarios:
        scen_df = df[df["scenario_id"] == scenario_id]
        
        with st.expander(f"📍 {scenario_id}", expanded=False):
            # 构建表格数据
            table_data = []
            for _, row in scen_df.iterrows():
                table_data.append({
                    "模式": row["mode"],
                    "Δ距离 (km)": f"{row['delta_dist_km']:.1f}",
                    "Δ距离 (%)": f"{row['rel_dist_pct']:.2f}%",
                    "Δ成本": f"{row['delta_cost']:.2f}",
                    "Δ成本 (%)": f"{row['rel_cost_pct']:.2f}%",
                    "风险下降 (%)": (
                        f"{row['risk_reduction_pct']:.2f}%"
                        if pd.notna(row["risk_reduction_pct"])
                        else "N/A"
                    ),
                })
            
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)


def render_scatter_plot(df: pd.DataFrame) -> None:
    """
    渲染散点图：距离增加% vs 风险下降%。
    
    Args:
        df: 评估结果 DataFrame
    """
    st.subheader("📈 距离增加 vs 风险下降")
    
    # 过滤有效数据
    valid_df = df[pd.notna(df["risk_reduction_pct"])].copy()
    
    if valid_df.empty:
        st.info("暂无有效的风险下降数据。")
        return
    
    # 创建散点图数据
    scatter_data = []
    for _, row in valid_df.iterrows():
        scatter_data.append({
            "scenario": row["scenario_id"],
            "mode": row["mode"],
            "distance_increase": row["rel_dist_pct"],
            "risk_reduction": row["risk_reduction_pct"],
        })
    
    df_scatter = pd.DataFrame(scatter_data)
    
    # 使用 Streamlit 的 scatter_chart
    try:
        import altair as alt
        
        # 创建 Altair 图表
        chart = (
            alt.Chart(df_scatter)
            .mark_circle(size=100, opacity=0.7)
            .encode(
                x=alt.X(
                    "distance_increase:Q",
                    title="距离增加 (%)",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "risk_reduction:Q",
                    title="风险下降 (%)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("mode:N", title="模式"),
                tooltip=["scenario:N", "mode:N", "distance_increase:Q", "risk_reduction:Q"],
            )
            .properties(
                width=600,
                height=400,
                title="各场景的距离-风险权衡",
            )
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # 添加参考线（Pareto 前沿）
        st.caption(
            "💡 **图表解读**: 左上角为最优（低距离增加、高风险下降）；"
            "右下角为次优（高距离增加、低风险下降）。"
        )
        
    except ImportError:
        # 备用：使用 Streamlit 内置的 scatter_chart
        st.scatter_chart(
            df_scatter,
            x="distance_increase",
            y="risk_reduction",
            color="mode",
        )


def render_summary_stats(summary: dict) -> None:
    """
    渲染总体统计信息。
    
    Args:
        summary: 全局总结字典
    """
    st.subheader("📊 全局统计")
    
    if not summary:
        st.info("暂无评估数据。")
        return
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    if "edl_safe" in summary:
        safe_stats = summary["edl_safe"]
        with col1:
            st.markdown("### EDL-Safe")
            st.metric(
                "平均风险下降",
                f"{safe_stats['avg_risk_reduction']:.1f}%",
            )
            st.metric(
                "平均距离增加",
                f"{safe_stats['avg_distance_increase']:.2f}%",
            )
            st.metric(
                "改善场景数",
                f"{safe_stats['scenarios_with_better_risk']}/{safe_stats['total_scenarios']}",
            )
    
    if "edl_robust" in summary:
        robust_stats = summary["edl_robust"]
        with col2:
            st.markdown("### EDL-Robust")
            st.metric(
                "平均风险下降",
                f"{robust_stats['avg_risk_reduction']:.1f}%",
            )
            st.metric(
                "平均距离增加",
                f"{robust_stats['avg_distance_increase']:.2f}%",
            )
            st.metric(
                "改善场景数",
                f"{robust_stats['scenarios_with_better_risk']}/{robust_stats['total_scenarios']}",
            )


def render() -> None:
    """
    渲染评估结果页面。
    """
    st.title("🔬 EDL 评估结果")
    st.caption("基于 eval_scenario_results 的多场景对比分析")
    
    # 加载数据
    df = load_eval_results()
    
    if df is None:
        st.warning(
            "⚠️ 评估结果文件不存在。\n\n"
            "请先运行：\n"
            "```bash\n"
            "python -m scripts.run_scenario_suite\n"
            "python -m scripts.eval_scenario_results\n"
            "```"
        )
        return
    
    # 生成全局总结
    summary = generate_global_summary(df)
    
    # 创建三个 Tab
    tab1, tab2, tab3 = st.tabs(["📋 详细数据", "📈 可视化分析", "📝 结论总结"])
    
    with tab1:
        render_scenario_table(df)
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_scatter_plot(df)
        
        with col2:
            render_summary_stats(summary)
    
    with tab3:
        conclusion_text = generate_conclusion_text(summary)
        st.markdown(conclusion_text)
        
        # 添加数据导出功能
        st.subheader("📥 导出数据")
        
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="下载评估结果 (CSV)",
            data=csv_bytes,
            file_name="eval_mode_comparison.csv",
            mime="text/csv",
        )



