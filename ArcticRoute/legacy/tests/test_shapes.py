from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.xfail(reason="Enable in P1/P2", strict=False)


def test_risk_env_shape_and_range() -> None:
    """风险场必须存在且在 [0, 1] 范围内，并且不能全部为 NaN。"""
    ds_path = Path(__file__).resolve().parents[1] / "data_processed" / "env_clean.nc"
    assert ds_path.exists(), f"缺少数据文件: {ds_path}"

    ds = xr.open_dataset(ds_path)
    try:
        assert "risk_env" in ds.data_vars, "risk_env 不存在于数据集中"
        arr = ds["risk_env"].values
        assert arr.size > 0, "risk_env 数据为空"
        assert not np.all(np.isnan(arr)), "risk_env 数据全部为 NaN"

        finite_arr = arr[np.isfinite(arr)]
        assert finite_arr.size > 0, "risk_env 全为 NaN 或无有效数据"
        assert finite_arr.min() >= -1e-6, f"风险值存在小于 0 的元素 {finite_arr.min()}"
        assert finite_arr.max() <= 1 + 1e-6, f"风险值存在大于 1 的元素 {finite_arr.max()}"
    finally:
        ds.close()
