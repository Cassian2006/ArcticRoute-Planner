from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arcticroute.core.edl_train_torch import train_edl_model_from_parquet


def _sample_existing_parquet(limit: int = 500) -> pd.DataFrame | None:
    pattern = Path("data_real/edl/training")
    candidates = sorted(pattern.glob("edl_dataset_*.parquet"))
    if not candidates:
        return None
    df = pd.read_parquet(candidates[0])
    return df.head(limit)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_torch_training_smoke(tmp_path: Path):
    df = _sample_existing_parquet()
    if df is None or df.empty:
        pytest.skip("No EDL parquet available for smoke test")

    sample_path = tmp_path / "sample.parquet"
    df.to_parquet(sample_path, index=False)

    config_path = tmp_path / "config.yaml"
    model_dir = tmp_path / "models"
    report_path = tmp_path / "report.json"

    sample_pattern = sample_path.as_posix()
    model_dir_str = model_dir.as_posix()
    report_path_str = report_path.as_posix()

    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  parquet_glob: \"{sample_pattern}\"",
                "  train_fraction: 0.7",
                "  random_seed: 0",
                "  target_column: \"label\"",
                "  feature_columns:",
                "    - sic",
                "    - wave_swh",
                "    - ais_density",
                "    - vessel_dwt",
                "    - vessel_max_ice_thickness",
                "model:",
                "  hidden_sizes: [8]",
                "  dropout: 0.0",
                "train:",
                "  batch_size: 64",
                "  num_epochs: 1",
                "  learning_rate: 1e-3",
                "  weight_decay: 0.0",
                "  device: \"cpu\"",
                "output:",
                f"  model_dir: \"{model_dir_str}\"",
                "  model_name: \"smoke.pt\"",
                f"  report_path: \"{report_path_str}\"",
            ]
        ),
        encoding="utf-8",
    )

    report = train_edl_model_from_parquet(config_path)

    model_file = model_dir / "smoke.pt"
    assert model_file.exists(), "Model file not created"
    assert report_path.exists(), "Report file not created"
    assert report.get("train_samples", 0) > 0
