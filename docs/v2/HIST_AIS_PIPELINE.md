# HIST_AIS_PIPELINE（历史航线 / 拥挤风险链路）

- 现状（旧版 risk_interact 来源）
  - 路径：ArcticRoute/data_processed/risk 下的 R_interact_YYYYMM.nc / risk_interact_YYYYMM.nc
  - 变量：risk / R_interact / risk_interact（随历史版本不一致）
  - 数据特征：通常为静态或基于启发式/事故数据推断的交互/拥挤风险，与真实 AIS 交通密度未严格对齐。

- 新设计目标
  - 让“拥挤风险”真正来自 AIS 交通密度，并与环境网格严格对齐。
  - 形成可追溯的月度 traffic_density 与归一化的 congestion_risk。

- 输入数据
  - 原始 AIS 轨迹：csv 或 parquet
  - 统一约定目录：ArcticRoute/data_raw/ais
  - 命名示例：ais_YYYYMM_*.parquet / ais_YYYYMM_*.csv
  - 字段要求：包含经纬度列（列名自动识别：lat/lon/Latitude/Longitude 等）

- 网格与对齐
  - 使用现有环境网格作为空间对齐标准：
    - 首选：ArcticRoute/data_processed/env/env_clean.nc（或 ArcticRoute/data_processed/env_clean.nc）
    - 兜底：ArcticRoute/config/grid_spec.json
  - 网格约定：
    - 维度：latitude（H）、longitude（W），1D 单调数组
    - 将 AIS 点按最近邻映射到该网格

- 统计与中间结果
  - 按月份聚合 traffic_density(ym, i, j)：
    - 基本版本：计数（AIS 点次数，近似“船舶次数/小时”）
    - 可扩展：若输入包含时间戳，可按“船舶小时”累计（后续迭代）
  - 过滤：丢弃 NaN、越界坐标

- 输出数据
  - 路径：ArcticRoute/data_processed/risk/traffic_density_YYYYMM.nc
  - 变量：
    - traffic_density：float32，维度 (latitude, longitude)
    - congestion_risk：float32，维度 (latitude, longitude)，由 traffic_density 派生
      - 计算：norm = (log1p(traffic_density) - min) / (max - min + 1e-6)，范围约 0..1

- 与 UI/风险层的关系
  - “拥挤风险”滑条作用于 key="interact" 所绑定的栅格。
  - 新接线：优先从 traffic_density_YYYYMM.nc 读取变量 congestion_risk；若不存在则回退旧版 risk_interact/R_interact。
  - 成本构建：在 build_cost_da 中，key="interact" 的层权重即 UI 滑条 w_interact。

- 兼容性与回退
  - 如果找不到 traffic_density_YYYYMM.nc 或其中不含 congestion_risk：
    - 回退到 R_interact_YYYYMM.nc / risk_interact_YYYYMM.nc 的旧逻辑。
  - 若网格不完全一致：在加载时插值到当前环境网格（与 newenv 插值逻辑保持一致）。

- 自测建议
  - 运行：python ArcticRoute/scripts/preprocess_ais_to_density.py --ym 202412
  - 启动应用，选择 202412：
    - “显示图层”中打开拥挤/traffic 图层，看是否有结构化热力
    - 调整“拥挤风险”滑条，观察路线对高 traffic 区域的响应（绕开/贴近取决于权重策略）

---

附：破冰船识别与“安全走廊字段”

- 识别破冰船（Icebreaker）：
  - 在 AIS 字段 ship_type / shiptype / type / vessel_type / ais_type 中模糊匹配包含 "icebreaker", "ice-break"（不区分大小写）
  - 若持有权威 IMO ShipTypeCode 映射，可扩展按代码识别（当前实现以文本匹配为主）
  - 也可维护一份已知破冰船列表（MMSI/IMO）作为白名单补充

- 预处理脚本：ArcticRoute/scripts/preprocess_icebreaker_corridors.py
  - 输入：ArcticRoute/data_raw/ais + ym（YYYYMM）
  - 筛选破冰船记录后，在 env 网格上统计密度并归一化（log1p + min-max）
  - 输出：ArcticRoute/data_processed/risk/icebreaker_corridor_{ym}.nc
    - 变量：icebreaker_corridor，维度 (latitude, longitude)，范围 0..1（越高表示越常被破冰船通行）

- 成本接线（折扣）：
  - 在成本构建（build_cost_da）中，若启用“破冰走廊折扣”且成功加载 corridor：
    - 对冰险成本应用折扣：effective_ice_cost = ice_cost * (1 - alpha * corridor)，并裁剪到非负
    - alpha 可由 UI 滑条或配置给定（例如 0.5/0.8）
  - 仅做第一版：只对冰险折扣，不叠加复杂先验
