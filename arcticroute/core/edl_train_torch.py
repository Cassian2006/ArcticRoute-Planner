from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import torch  # type: ignore
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except (ImportError, OSError) as e:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    TORCH_IMPORT_ERROR = e
else:
    TORCH_IMPORT_ERROR = None


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


@dataclass
class TrainConfig:
    parquet_glob: str
    train_fraction: float
    random_seed: int
    target_column: str
    feature_columns: list[str]
    hidden_sizes: list[int]
    dropout: float
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    device: str
    model_dir: Path
    model_name: str
    report_path: Path


def load_train_config(path: str | Path = "configs/edl_train.yaml") -> TrainConfig:
    cfg_path = Path(path)
    data = _load_yaml(cfg_path)
    data_cfg = data.get("data") or {}
    model_cfg = data.get("model") or {}
    train_cfg = data.get("train") or {}
    out_cfg = data.get("output") or {}

    feature_columns = data_cfg.get("feature_columns") or []
    if not feature_columns:
        raise ValueError("data.feature_columns must be non-empty")

    parquet_glob = data_cfg.get("parquet_glob")
    if not parquet_glob:
        raise ValueError("data.parquet_glob is required")

    target_column = data_cfg.get("target_column") or "label"

    return TrainConfig(
        parquet_glob=str(parquet_glob),
        train_fraction=float(data_cfg.get("train_fraction", 0.8)),
        random_seed=int(data_cfg.get("random_seed", 42)),
        target_column=target_column,
        feature_columns=[str(c) for c in feature_columns],
        hidden_sizes=[int(h) for h in model_cfg.get("hidden_sizes", [64, 64])],
        dropout=float(model_cfg.get("dropout", 0.1)),
        batch_size=int(train_cfg.get("batch_size", 512)),
        num_epochs=int(train_cfg.get("num_epochs", 5)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        device=str(train_cfg.get("device", "cpu")),
        model_dir=Path(out_cfg.get("model_dir") or "data_real/edl/models"),
        model_name=str(out_cfg.get("model_name") or "edl_torch_demo.pt"),
        report_path=Path(out_cfg.get("report_path") or "data_real/edl/reports/edl_train_report.json"),
    )


# ============================================================================
# PyTorch-dependent classes and functions (only defined if torch is available)
# ============================================================================

if torch is not None:

    class EDLDataset(Dataset):
        def __init__(self, parquet_glob: str, feature_columns: Sequence[str], target_column: str) -> None:
            files = [Path(p) for p in glob.glob(parquet_glob)]
            if not files:
                raise FileNotFoundError(f"No parquet files match pattern: {parquet_glob}")
            frames = []
            for f in files:
                try:
                    frames.append(pd.read_parquet(f))
                except Exception as exc:
                    raise RuntimeError(f"Failed to read parquet {f}: {exc}") from exc
            df = pd.concat(frames, ignore_index=True)

            available_cols = [c for c in feature_columns if c in df.columns]
            # Drop columns that are entirely NaN
            usable_cols = [c for c in available_cols if not df[c].isna().all()]
            if not usable_cols:
                raise ValueError("No usable feature columns found after dropping all-NaN columns")

            self.target_col = target_column
            if self.target_col not in df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in data")

            self.features = df[usable_cols]
            self.labels = df[self.target_col].astype(int)

            # Simple normalization: scale dwt and ais_density into reasonable ranges
            normed = self.features.copy()
            if "vessel_dwt" in normed.columns:
                normed["vessel_dwt"] = normed["vessel_dwt"] / 100000.0
            if "ais_density" in normed.columns:
                normed["ais_density"] = np.clip(normed["ais_density"], 0.0, 1.0)

            # Fill remaining NaNs with column mean (or 0 if mean is NaN)
            for col in normed.columns:
                mean_val = normed[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0.0
                normed[col] = normed[col].fillna(mean_val)

            self.feature_array = normed.astype(np.float32).values
            self.label_array = self.labels.values.astype(np.int64)
            self.columns = list(normed.columns)

        def __len__(self) -> int:
            return len(self.label_array)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = torch.from_numpy(self.feature_array[idx])
            y = torch.tensor(self.label_array[idx], dtype=torch.long)
            return x, y

    class SimpleEDLNet(nn.Module):
        def __init__(self, in_dim: int, hidden_sizes: List[int], dropout: float = 0.0) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev = in_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)

    def _split_dataset(dataset: EDLDataset, train_fraction: float, seed: int):
        train_len = int(len(dataset) * train_fraction)
        val_len = len(dataset) - train_len
        if train_len == 0 or val_len == 0:
            raise ValueError("Train/val split produced empty subset; adjust train_fraction or dataset size")
        generator = torch.Generator().manual_seed(seed)
        return random_split(dataset, [train_len, val_len], generator=generator)

    def _compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        acc = correct / total if total > 0 else 0.0
        return {"accuracy": acc}


def train_edl_model_from_parquet(config_path: str | Path = "configs/edl_train.yaml") -> Dict[str, Any]:
    if torch is None or nn is None:
        print("[EDL_TRAIN] PyTorch 在当前环境不可用，跳过训练。")
        if TORCH_IMPORT_ERROR is not None:
            print(f"[EDL_TRAIN] 详细错误: {TORCH_IMPORT_ERROR}")
        return {
            "status": "torch_unavailable",
            "detail": str(TORCH_IMPORT_ERROR),
        }

    cfg = load_train_config(config_path)

    dataset = EDLDataset(cfg.parquet_glob, cfg.feature_columns, cfg.target_column)
    train_ds, val_ds = _split_dataset(dataset, cfg.train_fraction, cfg.random_seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device(cfg.device)
    model = SimpleEDLNet(in_dim=len(dataset.columns), hidden_sizes=cfg.hidden_sizes, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history: list[dict[str, Any]] = []
    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += yb.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * yb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_correct / max(train_total, 1),
            "val_loss": val_loss / max(val_total, 1),
            "val_acc": val_correct / max(val_total, 1),
        }
        history.append(epoch_log)
        print(
            f"[EDL_TRAIN] epoch={epoch_log['epoch']} "
            f"train_loss={epoch_log['train_loss']:.4f} train_acc={epoch_log['train_acc']:.4f} "
            f"val_loss={epoch_log['val_loss']:.4f} val_acc={epoch_log['val_acc']:.4f}"
        )

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = cfg.model_dir / cfg.model_name
    torch.save({"model_state": model.state_dict(), "features": dataset.columns}, model_path)

    final_metrics = history[-1] if history else {}
    report: Dict[str, Any] = {
        "config_path": str(Path(config_path).resolve()),
        "parquet_glob": cfg.parquet_glob,
        "num_samples": len(dataset),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "features": dataset.columns,
        "history": history,
        "final": final_metrics,
        "model_path": str(model_path.resolve()),
        "report_path": str(cfg.report_path.resolve()),
    }
    import json

    cfg.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
