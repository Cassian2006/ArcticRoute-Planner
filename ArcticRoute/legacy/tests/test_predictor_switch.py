import sys
from pathlib import Path

import pytest
import xarray as xr

from ArcticRoute.api import cli
from ArcticRoute.core.predictors.cv_sat import SatCVPredictor
from ArcticRoute.core.predictors.dl_ice import DLIcePredictor
from ArcticRoute.core.predictors.env_nc import EnvNCPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.xfail(reason="Enable in P1/P2", strict=False)


@pytest.mark.skipif(sys.platform != "win32", reason="Placeholder predictors are only verified on Windows CI")
def test_predictor_switch_alignment():
    config = cli.load_yaml_file(PROJECT_ROOT / "config" / "runtime.yaml")
    data_cfg = config.get("data") or {}
    run_cfg = config.get("run") or {}

    env_token = data_cfg.get("env_nc")
    assert env_token, "data.env_nc must be configured"
    env_path = Path(env_token)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    var_name = run_cfg.get("var", "risk_env")
    tidx = int(run_cfg.get("tidx", 0))

    with xr.open_dataset(env_path) as ds:
        assert var_name in ds, f"{var_name} missing in {env_path}"
        template = ds[var_name].load()

    base_predictor = EnvNCPredictor(env_path, var_name)
    base_output = base_predictor.prepare(tidx)
    assert base_output.risk.shape == template.shape
    assert base_output.risk.dims == template.dims

    sat_predictor = SatCVPredictor(env_path, var_name)
    sat_ds = sat_predictor.prepare(tidx)
    assert "sat_dummy" in sat_ds
    assert sat_ds["sat_dummy"].shape == template.shape

    ice_predictor = DLIcePredictor(env_path, var_name)
    ice_ds = ice_predictor.prepare(tidx)
    assert "ice_prob" in ice_ds
    assert ice_ds["ice_prob"].shape == template.shape
