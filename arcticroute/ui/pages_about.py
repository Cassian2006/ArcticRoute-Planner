"""
关于页 - 项目信息和文档
"""

from __future__ import annotations

import streamlit as st


def render_about() -> None:
    """渲染关于页"""
    
    st.title("ℹ 关于 ArcticRoute")
    
    st.markdown("""
    ## 北极航线智能规划系统
    
    **ArcticRoute** 是一个基于多模态环境数据和机器学习的北极航线智能规划系统。
    
    ### 核心特性
    
    -  **多模态成本场**: 整合海冰、波浪、AIS、冰级等多源数据
    -  **EDL 风险评估**: 基于 Evidential Deep Learning 的不确定性感知风险评估
    -  **智能路径规划**: A* 算法 + PolarRoute 框架
    -  **可视化分析**: 交互式地图和成本分解图表
    -  **规则约束**: Polar Code + POLARIS 风险评估系统
    
    ### 技术栈
    
    **后端**
    - Python 3.11+
    - NumPy, Pandas, Xarray
    - NetCDF4, GeoPandas
    - PyTorch (可选，用于 EDL)
    
    **前端**
    - Streamlit
    - Pydeck (地图可视化)
    - Plotly (图表)
    - Altair (统计图表)
    
    ### 数据源
    
    **环境数据**
    - Copernicus Marine Service (CMEMS)
      - 海冰浓度 (SIC)
      - 海冰厚度 (SIT)
      - 有效波高 (SWH)
      - 海冰漂移速度
    
    **静态资产**
    - AIS 历史航迹数据
    - 全球港口数据库
    - 水深测量数据
    - 主航道走廊数据
    
    ### 引用
    
    如果您在研究中使用了 ArcticRoute，请引用：
    
    ```
    @software{arcticroute2024,
      title={ArcticRoute: 北极航线智能规划系统},
      author={Your Name},
      year={2024},
      url={https://github.com/yourusername/arcticroute}
    }
    ```
    
    ### 许可证
    
    本项目采用 MIT 许可证。详见 LICENSE 文件。
    
    ### 联系方式
    
    -[object Object].email@example.com
    -  GitHub: https://github.com/yourusername/arcticroute
    -  文档: https://arcticroute.readthedocs.io
    
    ---
    
    © 2024 ArcticRoute Project. All rights reserved.
    """)
    
    # 版本信息
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("版本", "2.0.0")
    
    with col2:
        st.metric("Python", "3.11+")
    
    with col3:
        st.metric("Streamlit", "1.28+")

