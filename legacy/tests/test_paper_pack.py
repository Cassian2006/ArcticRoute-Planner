from pathlib import Path
import json

from ArcticRoute.paper import figures as PFIG
from ArcticRoute.paper import tables as PTAB
from ArcticRoute.paper import bundle as PB


def test_figures_and_tables_min(tmp_path: Path):
    # 使用占位 month/scenario 生成最少资产
    ym = "202412"
    scen = "nsr_wbound"
    f1 = PFIG.fig_calibration(ym)
    f2 = PFIG.fig_pareto(ym, scen)
    t1 = PTAB.tab_metrics_summary(ym, [scen])
    assert f1.exists() and f2.exists(), "figures should be produced"
    assert t1.exists(), "table should be produced"
    # 检查 .meta.json 邻接
    assert Path(str(f1) + ".meta.json").exists()
    assert Path(str(t1) + ".meta.json").exists()


def test_bundle_and_check(tmp_path: Path):
    # 确保 reports/paper 下至少有一个文件
    ym = "202412"
    scen = "nsr_wbound"
    PFIG.fig_calibration(ym)
    PTAB.tab_metrics_summary(ym, [scen])
    out = PB.build_bundle("quick", tag="test")
    zip_path = Path(out["zip"]) if isinstance(out, dict) else Path(out["zip"])  # type: ignore[index]
    assert zip_path.exists(), "zip should exist"
    res = PB.check_bundle(zip_path)
    assert res.get("ok", False), f"bundle check failed: {res}"






