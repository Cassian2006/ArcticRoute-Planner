# ECO 模块说明与接入（newenv 驱动）

- 现状（1.0 基线）
  - 总结函数 summarize_route 仅基于距离给出估算燃油（estimated_fuel_ton = 距离 × 系数）。
  - evaluate_route_eco 优先调用旧版 eco 模块（fuel_per_nm_map + route_eval），不可用时回退为距离估算。

- 目标（本阶段）
  - 接入 newenv 中的物理场（sic/sithick/wave_swh）计算“环境修正后的燃油消耗”。
  - 暂不改变 A* 代价（不反向影响路径），先在路线摘要中量化展示（燃油/CO₂）。
  - 后续阶段可把“预期燃油消耗”折算为代价项影响搜索。

- 输入数据（来自 newenv）
  - ice_copernicus_sic.nc → 变量 sic（0..1，海冰浓度）
  - ice_copernicus_sithick.nc → 变量 sithick（m，冰厚；若缺失则忽略）
  - wave_swh.nc → 变量 wave_swh（m，显著波高）
  - 以上均在 ArcticRoute/data_processed/newenv 下，由 newenv_loader 统一加载并插值到环境网格。

- 简化燃油修正模型
  - 设基准单位距离油耗 base_fuel_per_km = 常数（吨/km），按船型/吨位可调整（后续引入 vessel_profile）。
  - 对路径每一步（栅格边）计算：
    - 取该格点的 sic, sithick, wave_swh（近邻/对齐网格）。
    - 冰修正：f_ice = 1 + a1 * sic^2 + a2 * sithick
    - 浪修正：f_wave = 1 + b1 * max(0, wave_swh - 1.0)
    - 总系数 f = f_ice * f_wave
    - step_fuel = base_fuel_per_km * step_distance_km * f
  - 总燃油 fuel_tons = Σ step_fuel
  - 平静水面燃油 base_fuel_tons = Σ (base_fuel_per_km * step_distance_km)
  - CO₂ 折算：co2_tons = fuel_tons × 3.114（可按实际因子调整）
  - 参数建议（可调整）：a1=1.2，a2=0.2，b1=0.12；wave_swh 阈值 1.0 m

- 接口与实现
  - newenv_loader.load_newenv_for_eco(ym, env_lat, env_lon) → dict
    - 返回已对齐到环境网格的 DataArray：{"sic":..., "sithick":..., "wave_swh":...}
  - core/eco/eco_newenv.compute_fuel_along_route(path_ij, env_lat, env_lon, newenv_for_eco, vessel_profile) → dict
    - 返回 {"fuel_tons", "co2_tons", "base_fuel_tons"}
  - planner_service.evaluate_route_eco 优先调用上述 newenv 模型；不可用时回退旧 eco

- 展示与校验
  - UI 路线摘要卡片显示：燃油 (t)、CO₂ (t)；Eco 不可用时回退距离估算。
  - 自测：切换不同月份/路径，观察在高 SIC/SWH 区域燃油上升；在平静水面接近 base_fuel_tons。
