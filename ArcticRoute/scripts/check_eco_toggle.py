"""ECO 开关体检脚本：在同一条已计算好的路线结果上，分别做 simple 与 model 评估，
并打印清晰的对比，验证 Eco 开关对数值是否有真实影响。

运行方式（仓库根目录）：
    python ArcticRoute/scripts/check_eco_toggle.py
"""

import sys
from pathlib import Path

# -- Add project root to sys.path --
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# -- End of path setup --

from ArcticRoute.core import planner_service


def main():
    ym = "202412"
    start_ij = (60, 150)
    end_ij = (60, 1000)

    print("=== ECO toggle diagnostic ===")

    # 1) 加载环境 + 计算一次路线（A* 只跑一次）
    env = planner_service.load_environment(ym=ym, w_ice=0.7, w_accident=0.2)
    route = planner_service.compute_route(env, start_ij, end_ij, True, "manhattan")
    if not route.reachable:
        print("route unreachable; 请检查起终点/成本场数据是否有效")
        return

    # 路线摘要
    summ = planner_service.summarize_route(route)
    print(f"distance_km: {summ['distance_km']}")

    # 2) 对同一条路线做两种 ECO 评估
    eco_simple = planner_service.estimate_eco_simple(route)
    eco_model = planner_service.evaluate_route_eco(route, env)

    # 判定 model 的模式（真模型 or 回退）
    ok_model = bool(eco_model.details.get("ok", False))
    mode_model = "eco_model" if ok_model else "fallback_simple"

    # 打印
    print(f"simple: fuel={eco_simple.fuel_total_t} t, co2={eco_simple.co2_total_t} t, mode=\"distance_based\"")
    if not ok_model:
        reason = eco_model.details.get("reason") or eco_model.details.get("error", "")
        print(f"model:  fuel={eco_model.fuel_total_t} t, co2={eco_model.co2_total_t} t, mode=\"{mode_model}\" (reason={reason})")
        print("说明：当前已回退到 simple，因此与 simple 结果相同或接近。")
    else:
        print(f"model:  fuel={eco_model.fuel_total_t} t, co2={eco_model.co2_total_t} t, mode=\"{mode_model}\"")

    # Diff
    dfuel = round(eco_model.fuel_total_t - eco_simple.fuel_total_t, 2)
    dco2 = round(eco_model.co2_total_t - eco_simple.co2_total_t, 2)
    print(f"delta:  fuel_diff={dfuel} t, co2_diff={dco2} t")


if __name__ == "__main__":
    main()





