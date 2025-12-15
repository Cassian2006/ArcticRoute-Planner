"""
POLARIS 集成示例

演示如何在路由规划中使用 POLARIS 约束。
"""

import numpy as np
from arcticroute.core.constraints.polaris import compute_rio_for_cell, OperationLevel


def example_1_basic_rio_calculation():
    """示例 1: 基本 RIO 计算"""
    print("=" * 60)
    print("示例 1: 基本 RIO 计算")
    print("=" * 60)
    
    # 场景：PC4 船舶在北极
    scenarios = [
        {"sic": 0.0, "thickness": 0.0, "desc": "开水"},
        {"sic": 0.3, "thickness": 0.2, "desc": "轻冰"},
        {"sic": 0.6, "thickness": 0.4, "desc": "中冰"},
        {"sic": 0.9, "thickness": 1.0, "desc": "重冰"},
    ]
    
    ice_class = "PC4"
    
    for scenario in scenarios:
        meta = compute_rio_for_cell(
            sic=scenario["sic"],
            thickness_m=scenario["thickness"],
            ice_class=ice_class
        )
        
        print(f"\n{scenario['desc']}:")
        print(f"  SIC: {scenario['sic']:.1%}, 厚度: {scenario['thickness']:.2f}m")
        print(f"  冰型: {meta.ice_type}")
        print(f"  RIO: {meta.rio:.1f}")
        print(f"  操作等级: {meta.level}")
        if meta.speed_limit_knots:
            print(f"  速度限制: {meta.speed_limit_knots} knots")


def example_2_ice_class_comparison():
    """示例 2: 不同冰级的比较"""
    print("\n" + "=" * 60)
    print("示例 2: 不同冰级的比较")
    print("=" * 60)
    
    # 固定条件
    sic = 0.7
    thickness = 0.5
    
    ice_classes = ["PC1", "PC3", "PC5", "PC7"]
    
    print(f"\n固定条件: SIC={sic:.0%}, 厚度={thickness}m")
    print(f"\n{'冰级':<8} {'RIO':<8} {'操作等级':<12} {'速度限制':<15}")
    print("-" * 50)
    
    for ice_class in ice_classes:
        meta = compute_rio_for_cell(sic=sic, thickness_m=thickness, ice_class=ice_class)
        speed_str = f"{meta.speed_limit_knots} knots" if meta.speed_limit_knots else "无限制"
        print(f"{ice_class:<8} {meta.rio:<8.1f} {meta.level:<12} {speed_str:<15}")


def example_3_cost_grid_integration():
    """示例 3: 成本网格集成"""
    print("\n" + "=" * 60)
    print("示例 3: 成本网格集成（POLARIS 约束）")
    print("=" * 60)
    
    # 创建模拟网格
    ny, nx = 5, 5
    sic_grid = np.random.uniform(0.2, 0.9, (ny, nx))
    thickness_grid = np.random.uniform(0.1, 1.5, (ny, nx))
    
    ice_class = "PC5"
    cost_grid = np.ones((ny, nx)) * 100  # 基础成本
    
    print(f"\n网格大小: {ny} x {nx}")
    print(f"船舶冰级: {ice_class}")
    
    # 应用 POLARIS 约束
    special_count = 0
    elevated_count = 0
    normal_count = 0
    
    for i in range(ny):
        for j in range(nx):
            meta = compute_rio_for_cell(
                sic=sic_grid[i, j],
                thickness_m=thickness_grid[i, j],
                ice_class=ice_class
            )
            
            if meta.level == "special":
                cost_grid[i, j] = 1e10  # Hard block
                special_count += 1
            elif meta.level == "elevated":
                cost_grid[i, j] *= 2.0  # Soft penalty
                elevated_count += 1
            else:
                normal_count += 1
    
    print(f"\n约束应用结果:")
    print(f"  Normal (无限制): {normal_count} 个网格")
    print(f"  Elevated (速度限制): {elevated_count} 个网格")
    print(f"  Special (不可达): {special_count} 个网格")
    print(f"\n成本网格统计:")
    print(f"  最小成本: {cost_grid[cost_grid < 1e10].min():.1f}")
    print(f"  最大成本: {cost_grid[cost_grid < 1e10].max():.1f}")
    print(f"  被阻挡网格: {np.sum(cost_grid >= 1e10)}")


def example_4_decayed_ice_table():
    """示例 4: 衰减冰条件"""
    print("\n" + "=" * 60)
    print("示例 4: 衰减冰条件 (Table 1.4)")
    print("=" * 60)
    
    sic = 0.6
    thickness = 0.4
    ice_class = "PC4"
    
    # 标准条件
    meta_std = compute_rio_for_cell(
        sic=sic,
        thickness_m=thickness,
        ice_class=ice_class,
        use_decayed_table=False
    )
    
    # 衰减条件
    meta_decayed = compute_rio_for_cell(
        sic=sic,
        thickness_m=thickness,
        ice_class=ice_class,
        use_decayed_table=True
    )
    
    print(f"\n条件: SIC={sic:.0%}, 厚度={thickness}m, 冰级={ice_class}")
    print(f"\n{'条件':<15} {'RIO':<8} {'操作等级':<12} {'使用表':<15}")
    print("-" * 50)
    print(f"{'标准':<15} {meta_std.rio:<8.1f} {meta_std.level:<12} {meta_std.riv_used:<15}")
    print(f"{'衰减':<15} {meta_decayed.rio:<8.1f} {meta_decayed.level:<12} {meta_decayed.riv_used:<15}")
    
    if meta_std.rio != meta_decayed.rio:
        print(f"\n差异: RIO 相差 {abs(meta_std.rio - meta_decayed.rio):.1f}")


def main():
    """运行所有示例"""
    example_1_basic_rio_calculation()
    example_2_ice_class_comparison()
    example_3_cost_grid_integration()
    example_4_decayed_ice_table()
    
    print("\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()


