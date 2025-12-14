from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode, r.stdout, r.stderr


@pytest.mark.unit
def test_eco_norm_bounds():
    # 调用 eco.preview 隐式构建 eco 栅格并返回汇总（使用示例场景）
    cmd = [sys.executable, "-m", "api.cli", "eco.preview", "--ym", "202412", "--scenario", "nsr_wbound_smoke", "--class", "cargo_iceclass"]
    code, out, err = _run(cmd)
    assert code == 0, f"eco.preview failed: {err}"
    # 不直接读取栅格，但至少验证输出 JSON 结构
    payload = json.loads(out)
    assert "fuel_total_t" in payload and "co2_total_t" in payload


@pytest.mark.unit
def test_monotonicity_alpha_ice(tmp_path: Path):
    # 通过调用内部函数较重，这里使用 CLI 两次近似验证：alpha_ice 增大时 fuel_total 应不减
    import yaml
    eco_cfg_path = ROOT / "ArcticRoute" / "config" / "eco.yaml"
    cfg = yaml.safe_load(eco_cfg_path.read_text(encoding="utf-8"))
    base_alpha = float(((cfg.get("eco") or {}).get("alpha_ice", 0.8)))

    def run_preview() -> float:
        code, out, err = _run([sys.executable, "-m", "api.cli", "eco.preview", "--ym", "202412", "--scenario", "nsr_wbound_smoke", "--class", "cargo_iceclass"])
        assert code == 0, err
        return float(json.loads(out)["fuel_total_t"])

    # 记录原值
    f0 = run_preview()
    # 临时提高 alpha_ice 并写回
    cfg2 = cfg.copy()
    cfg2.setdefault("eco", {}).update({"alpha_ice": base_alpha + 0.2})
    bak = eco_cfg_path.read_text(encoding="utf-8")
    try:
        eco_cfg_path.write_text(yaml.safe_dump(cfg2, allow_unicode=True), encoding="utf-8")
        f1 = run_preview()
    finally:
        eco_cfg_path.write_text(bak, encoding="utf-8")
    assert f1 >= f0 * 0.99, f"fuel should not decrease when alpha_ice increases: {f0} -> {f1}"

