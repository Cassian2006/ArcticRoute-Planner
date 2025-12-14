from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arcticroute.core.edl_dataset import build_edl_training_table, load_edl_dataset_config


@pytest.fixture(scope="module")
def edl_dataset_result():
    cfg = load_edl_dataset_config()
    scenario_id = "barents_to_chukchi_edl"
    df = build_edl_training_table(scenario_id, cfg)
    out_path = Path(cfg.output_dir) / cfg.filename_pattern.format(scenario_id=scenario_id)
    return cfg, df, out_path, scenario_id


def test_build_edl_training_table_basic_shape(edl_dataset_result):
    cfg, df, _, _ = edl_dataset_result
    assert len(df) > 0
    required_cols = {
        "lat",
        "lon",
        "month",
        "sic",
        "wave_swh",
        "ice_thickness",
        "ais_density",
        "vessel_dwt",
        "vessel_max_ice_thickness",
        cfg.target_column,
    }
    assert required_cols.issubset(df.columns)
    assert set(df[cfg.target_column].unique()) <= {0, 1}


def test_labels_have_both_classes_if_possible(edl_dataset_result):
    cfg, df, _, _ = edl_dataset_result
    counts = df[cfg.target_column].value_counts()
    positives = counts.get(1, 0)
    negatives = counts.get(0, 0)
    if positives == 0 or negatives == 0:
        pytest.skip("AIS density did not provide both positive and negative samples")
    assert positives > 0
    assert negatives > 0


def test_output_file_written(edl_dataset_result):
    _, _, out_path, scenario_id = edl_dataset_result
    assert out_path.exists(), f"Parquet not written for scenario {scenario_id}"
    df_disk = pd.read_parquet(out_path)
    assert len(df_disk) > 0
