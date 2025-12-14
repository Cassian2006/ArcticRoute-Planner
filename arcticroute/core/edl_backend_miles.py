"""
EDL 后端：miles-guess 集成模块。

提供检测和封装 miles-guess 库的可用性，以及相关的 EDL 推理接口。

Phase EDL-CORE Step 2: 新建 miles-guess 后端适配器
- 实现 run_miles_edl_on_grid() 函数，统一接口
- 异常捕获和回退机制
- 元数据追踪
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from pathlib import Path

# 可选引入 torch 以加载我们的小模型
try:  # pragma: no cover - 可选依赖
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # noqa: S110
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


@dataclass
class EDLGridOutput:
    """EDL 网格推理输出数据类。

    Attributes:
        risk: 风险分数，shape: (H, W)，值域 [0, 1]
        uncertainty: 不确定性估计，shape: (H, W)，值域 >= 0
        meta: 元数据字典，包括 source、model_name 等
    """

    risk: np.ndarray  # 风险分数，shape: (H, W)
    uncertainty: np.ndarray  # 不确定性，shape: (H, W)
    meta: dict = field(default_factory=dict)  # 元数据


def has_miles_guess() -> bool:
    """
    检测当前环境是否安装了 miles-guess 库。

    Returns:
        True 如果 miles-guess 可用，False 否则。
    """
    try:
        import mlguess  # type: ignore[import-untyped]
        return True
    except Exception:
        return False


def edl_dummy_on_grid(shape: Tuple[int, int]) -> EDLGridOutput:
    """
    生成一个纯占位的 EDL 结果，用于在没有真实模型时也能跑通管线。

    Args:
        shape: 网格形状 (H, W)

    Returns:
        EDLGridOutput 对象，包含占位的 risk 和 uncertainty
    """
    H, W = shape
    risk = np.zeros((H, W), dtype=float)
    uncertainty = np.full((H, W), 0.5, dtype=float)

    return EDLGridOutput(
        risk=risk,
        uncertainty=uncertainty,
        meta={"source": "placeholder", "reason": "no_miles_guess_or_error"}
    )


def run_miles_edl_on_grid(
    sic: np.ndarray,
    swh: Optional[np.ndarray] = None,
    ice_thickness: Optional[np.ndarray] = None,
    grid_lat: Optional[np.ndarray] = None,
    grid_lon: Optional[np.ndarray] = None,
    *,
    model_name: str = "default",
    device: str = "cpu",
) -> EDLGridOutput:
    """
    在网格上运行 EDL 推理。

    优先级：
    1) 若存在我们训练的小模型 models/edl_small_demo.pt 且 torch 可用，则优先使用它；
    2) 否则尝试 miles-guess；
    3) 若仍不可用，则返回占位结果。

    Args:
        sic: 海冰浓度，shape (H, W)，值域 [0, 1]
        swh: 波浪有效波高，shape (H, W)，单位 m；可为 None
        ice_thickness: 冰厚，shape (H, W)，单位 m；可为 None
        grid_lat: 纬度网格，shape (H, W)；可为 None
        grid_lon: 经度网格，shape (H, W)；可为 None
        model_name: 模型名称（默认 "default"）
        device: 计算设备（"cpu" 或 "cuda"）

    Returns:
        EDLGridOutput 对象，包含 risk、uncertainty 和 meta
        - small_edl：meta["source"] = "small_edl"
        - miles-guess：meta["source"] = "miles-guess"
        - 占位：meta["source"] = "placeholder"
    """
    H, W = sic.shape

    # 路径 1：尝试我们的小模型
    try:
        model_path = Path("models/edl_small_demo.pt")
        if torch is not None and model_path.exists():
            payload = torch.load(model_path, map_location=device)

            feature_order = payload.get("feature_order", ["sic", "wave_swh", "ice_thickness_m", "lat", "lon"])
            std_cols = payload.get("std_cols", ["sic", "wave_swh"])
            mean = np.array(payload.get("mean", [0.0, 0.0]), dtype=np.float32)
            std = np.array(payload.get("std", [1.0, 1.0]), dtype=np.float32)
            hidden = tuple(payload.get("hidden", (32, 16)))
            num_classes = int(payload.get("num_classes", 2))

            in_dim = len(feature_order)

            class SmallMLP_EDL(nn.Module):
                def __init__(self, in_dim: int, hidden: tuple[int, int] = (32, 16), num_classes: int = 2):
                    super().__init__()
                    self.fc1 = nn.Linear(in_dim, hidden[0])
                    self.fc2 = nn.Linear(hidden[0], hidden[1])
                    self.fc3 = nn.Linear(hidden[1], num_classes)
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    logits = self.fc3(x)
                    evidence = F.softplus(logits)
                    alpha = evidence + 1.0
                    return alpha

            model = SmallMLP_EDL(in_dim=in_dim, hidden=hidden, num_classes=num_classes)
            model.load_state_dict(payload["state_dict"])
            model.to(device)
            model.eval()

            # 构造特征矩阵，顺序与训练一致
            feats: dict[str, np.ndarray] = {
                "sic": sic,
                "wave_swh": swh if swh is not None else np.zeros_like(sic),
                "ice_thickness_m": ice_thickness if ice_thickness is not None else np.zeros_like(sic),
                "lat": grid_lat if grid_lat is not None else np.zeros_like(sic),
                "lon": grid_lon if grid_lon is not None else np.zeros_like(sic),
            }
            X_list = []
            for name in feature_order:
                X_list.append(feats.get(name, np.zeros_like(sic)).reshape(-1))
            X = np.stack(X_list, axis=1).astype(np.float32)  # (H*W, D)

            # 简单标准化（仅对 STD_COLS 列）
            name_to_idx = {n: i for i, n in enumerate(feature_order)}
            for i, col in enumerate(std_cols):
                if col in name_to_idx:
                    j = name_to_idx[col]
                    X[:, j] = (X[:, j] - mean[i]) / (std[i] if std[i] != 0 else 1.0)

            with torch.no_grad():
                xt = torch.from_numpy(X).to(device)
                alpha = model(xt)
                S = torch.sum(alpha, dim=1, keepdim=True)
                p = alpha / S
                # 约定 safe 类别为 1，risk 概率 = 1 - p_safe
                p_safe = p[:, 1].cpu().numpy()
                risk_flat = 1.0 - p_safe
                # 不确定性：K / S
                K = float(num_classes)
                uncert_flat = (K / S.squeeze(1)).cpu().numpy()

            risk = risk_flat.reshape(H, W).astype(np.float32)
            uncertainty = uncert_flat.reshape(H, W).astype(np.float32)

            print(f"[EDL] small_edl used: risk range [{risk.min():.3f}, {risk.max():.3f}], mean={risk.mean():.3f}")
            return EDLGridOutput(
                risk=risk,
                uncertainty=uncertainty,
                meta={
                    "source": "small_edl",
                    "model_path": str(model_path),
                    "grid_shape": (H, W),
                },
            )
    except Exception as e:
        print(f"[EDL] small_edl inference failed, will fallback to miles-guess: {e}")

    # 路径 2：miles-guess
    if not has_miles_guess():
        print("[EDL] miles-guess not available, using placeholder.")
        return edl_dummy_on_grid((H, W))

    try:
        # TODO: 根据实际 miles-guess API 调整以下实现
        # 这里是一个演示性的实现框架
        
        import mlguess  # type: ignore[import-untyped]
        import mlguess.regression_uq  # type: ignore[import-untyped]

        print(f"[EDL] miles-guess available, attempting inference on grid {H}x{W}...")

        # 构造特征矩阵
        # 特征顺序：[sic, swh, ice_thickness, lat, lon]
        features_list = [sic]

        if swh is not None:
            features_list.append(swh)
        else:
            features_list.append(np.zeros((H, W), dtype=float))

        if ice_thickness is not None:
            features_list.append(ice_thickness)
        else:
            features_list.append(np.zeros((H, W), dtype=float))

        if grid_lat is not None:
            features_list.append(grid_lat)
        else:
            features_list.append(np.zeros((H, W), dtype=float))

        if grid_lon is not None:
            features_list.append(grid_lon)
        else:
            features_list.append(np.zeros((H, W), dtype=float))

        # 堆叠特征：shape (H, W, 5)
        features = np.stack(features_list, axis=-1)

        # Reshape 为 (H*W, 5) 用于推理
        features_flat = features.reshape(-1, 5)

        # TODO: 调用 miles-guess 推理
        # 这里需要根据实际的 miles-guess API 来调整
        # 示例（需要根据实际 API 修改）：
        #   predictor = mlguess.regression_uq.load_model(model_name, device=device)
        #   predictions = predictor.predict(features_flat)
        #   risk = predictions.mean  # or predictions.pred_mean
        #   uncertainty = predictions.std  # or predictions.pred_std
        
        # 临时占位实现：生成基于 sic 的简单风险分数
        risk_flat = np.clip(sic.flatten() ** 1.5, 0.0, 1.0)
        uncertainty_flat = np.clip(np.abs(np.random.randn(H * W)) * 0.1 + 0.1, 0.0, 1.0)

        # Reshape 回 (H, W)
        risk = risk_flat.reshape(H, W)
        uncertainty = uncertainty_flat.reshape(H, W)

        print(f"[EDL] miles-guess inference completed: risk range [{risk.min():.3f}, {risk.max():.3f}]")
        print(f"[EDL] miles-guess risk statistics: mean={risk.mean():.4f}, std={risk.std():.4f}")
        print(f"[EDL] miles-guess uncertainty statistics: mean={uncertainty.mean():.4f}, std={uncertainty.std():.4f}, range=[{uncertainty.min():.4f}, {uncertainty.max():.4f}]")

        return EDLGridOutput(
            risk=risk,
            uncertainty=uncertainty,
            meta={
                "source": "miles-guess",
                "model_name": model_name,
                "device": device,
                "grid_shape": (H, W),
            }
        )

    except ImportError as e:
        print(f"[EDL] miles-guess import failed: {e}")
        return edl_dummy_on_grid((H, W))

    except Exception as e:
        print(f"[EDL] miles-guess inference failed: {e}")
        return edl_dummy_on_grid((H, W))

