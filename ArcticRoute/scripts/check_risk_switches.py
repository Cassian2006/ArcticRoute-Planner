"""高级风险参数开关体检脚本

验证 fusion_mode / w_interact / use_escort / risk_agg_mode 的有效性与优雅降级。

运行方式（仓库根目录）：
    python ArcticRoute/scripts/check_risk_switches.py
"""

import sys
from pathlib import Path

# -- Add project root to sys.path --
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# -- End of path setup --

from ArcticRoute.core import planner_service
import numpy as np


def _cost_stats(da):
    if da is None:
        return {"min": None, "max": None, "finite": 0}
    arr = np.asarray(da.values)
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        v = arr[finite_mask]
        return {"min": float(np.min(v)), "max": float(np.max(v)), "finite": int(finite_mask.sum())}
    return {"min": None, "max": None, "finite": 0}


def run_case(name: str, **kwargs):
    print(f"\n=== CASE: {name} ===")
    env = planner_service.load_environment(
        ym="202412",
        w_ice=0.7,
        w_accident=0.2,
        **kwargs,
    )
    stats = _cost_stats(env.cost_da)
    print(
        "fusion_mode_effective=", env.fusion_mode_effective,
        "| escort_applied=", env.escort_applied,
        "| w_interact=", env.w_interact,
        "| risk_agg_mode_effective=", env.risk_agg_mode_effective,
        "| risk_agg_alpha=", env.risk_agg_alpha,
    )
    print(f"cost field: min={stats['min']}, max={stats['max']}, finite_count={stats['finite']}")


def main():
    # 固定场景参数（起止点不需要，因为这里只看环境与成本场是否正常）
    run_case("baseline")
    run_case("fusion_linear", fusion_mode="linear")
    run_case("interact_on", w_interact=0.8)
    run_case("use_escort", use_escort=True)
    run_case("agg_cvar_0.9", fusion_mode="linear", risk_agg_mode="cvar", risk_agg_alpha=0.9)


if __name__ == "__main__":
    main()





