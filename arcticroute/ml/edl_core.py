"""
Evidential Deep Learning (EDL) 风险估计模块。

提供轻量、无外部训练依赖的 EDL 封装，用于海冰/航线风险评估。

核心思想：
  - 通过 Dirichlet 分布参数化不确定性
  - 输入特征 -> logits -> evidence -> alpha (Dirichlet 参数)
  - 期望概率 p = alpha / alpha.sum()
  - 不确定性 u = K / alpha.sum()

当 PyTorch 不可用时，提供 fallback 占位实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    # 当 PyTorch 不可用时，定义占位符以避免 NameError
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class EDLConfig:
    """EDL 配置数据类。

    Attributes:
        num_classes: 风险分类数（默认 3：safe/medium/high）
    """

    num_classes: int = 3


@dataclass
class EDLGridOutput:
    """EDL 推理输出数据类。

    Attributes:
        risk_mean: 期望风险分数，shape (H, W)，值域 [0, 1]
        uncertainty: 不确定性估计，shape (H, W)，值域 >= 0
    """

    risk_mean: np.ndarray  # shape (H, W), dtype float
    uncertainty: np.ndarray  # shape (H, W), dtype float


if TORCH_AVAILABLE:
    class EDLModel(nn.Module):  # type: ignore[misc,valid-type]
        """极简 EDL 模型：MLP + Dirichlet 头。

        仅用于推理，不包含训练逻辑。

        Attributes:
            input_dim: 输入特征维度
            num_classes: 风险分类数
            fc1, fc2, fc3: 线性层
        """

        def __init__(self, input_dim: int, num_classes: int = 3):
            """
            初始化 EDL 模型。

            Args:
                input_dim: 输入特征维度
                num_classes: 风险分类数（默认 3）
            """
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes

            # 极简 MLP：input_dim -> 16 -> 8 -> num_classes
            self.fc1 = nn.Linear(input_dim, 16)  # type: ignore[attr-defined]
            self.fc2 = nn.Linear(16, 8)  # type: ignore[attr-defined]
            self.fc3 = nn.Linear(8, num_classes)  # type: ignore[attr-defined]

            # 初始化权重（简单初始化，无需训练）
            self._init_weights()

        def _init_weights(self) -> None:
            """简单的权重初始化。"""
            for module in [self.fc1, self.fc2, self.fc3]:
                nn.init.xavier_uniform_(module.weight)  # type: ignore[attr-defined]
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # type: ignore[attr-defined]

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            """
            前向传播：输入特征 -> logits。

            Args:
                x: 输入张量，shape (..., input_dim)

            Returns:
                logits 张量，shape (..., num_classes)
            """
            x = F.relu(self.fc1(x))  # type: ignore[attr-defined]
            x = F.relu(self.fc2(x))  # type: ignore[attr-defined]
            logits = self.fc3(x)
            return logits

        def compute_edl_outputs(
            self, logits: torch.Tensor  # type: ignore[name-defined]
        ) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[name-defined]
            """
            从 logits 计算 EDL 输出：期望概率和不确定性。

            公式：
              - evidence = softplus(logits)
              - alpha = evidence + 1
              - p = alpha / alpha.sum(dim=-1, keepdim=True)
              - u = K / alpha.sum(dim=-1)

            Args:
                logits: logits 张量，shape (..., num_classes)

            Returns:
                (p, u) 元组，其中：
                  - p: 期望概率，shape (..., num_classes)
                  - u: 不确定性，shape (...)
            """
            # evidence = softplus(logits)
            evidence = F.softplus(logits)  # type: ignore[attr-defined]

            # alpha = evidence + 1
            alpha = evidence + 1.0

            # p = alpha / alpha.sum(dim=-1, keepdim=True)
            alpha_sum = alpha.sum(dim=-1, keepdim=True)
            p = alpha / alpha_sum

            # u = K / alpha.sum(dim=-1)
            K = self.num_classes
            u = K / alpha_sum.squeeze(-1)

            return p, u
else:
    # 当 PyTorch 不可用时，定义占位符类
    class EDLModel:  # type: ignore[no-redef]
        """占位符 EDL 模型（PyTorch 不可用时）。"""

        def __init__(self, input_dim: int, num_classes: int = 3):
            """初始化占位符模型。"""
            self.input_dim = input_dim
            self.num_classes = num_classes


def run_edl_on_features(
    features: np.ndarray,
    config: Optional[EDLConfig] = None,
) -> EDLGridOutput:
    """
    在特征网格上运行 EDL 推理。

    若 TORCH_AVAILABLE=False，返回占位符输出（risk_mean=0, uncertainty=1）。
    若 TORCH_AVAILABLE=True，使用极简 MLP + Dirichlet 头进行推理。

    异常处理：
    - 若推理过程中发生异常，捕获并打印错误信息，返回占位符输出
    - 不向上层抛出异常，确保管线可以平滑回退

    Args:
        features: 特征数组，shape (H, W, F)，其中：
                  - H, W: 网格高度和宽度
                  - F: 特征维度（例如 [sic, wave_swh, ice_thickness, lat, lon]）
        config: EDLConfig 对象，若为 None 则使用默认配置

    Returns:
        EDLGridOutput 对象，包含 risk_mean 和 uncertainty
    """
    if config is None:
        config = EDLConfig()

    H, W, F = features.shape

    # 若 PyTorch 不可用，返回占位符
    if not TORCH_AVAILABLE:
        print("[EDL][torch] PyTorch not available; using fallback constant risk.")
        risk_mean = np.zeros((H, W), dtype=float)
        uncertainty = np.ones((H, W), dtype=float)
        return EDLGridOutput(risk_mean=risk_mean, uncertainty=uncertainty)

    try:
        # 将特征转换为 PyTorch 张量
        # 形状：(H, W, F) -> (H*W, F)
        features_flat = features.reshape(-1, F)
        features_tensor = torch.from_numpy(features_flat).float()  # type: ignore[name-defined]

        # 创建 EDL 模型
        model = EDLModel(input_dim=F, num_classes=config.num_classes)
        model.eval()

        # 前向推理
        with torch.no_grad():  # type: ignore[name-defined]
            logits = model(features_tensor)  # shape (H*W, num_classes)
            p, u = model.compute_edl_outputs(logits)  # p: (H*W, num_classes), u: (H*W,)

            # 提取风险分数（假设类别顺序为 safe/medium/high，取 high 类的概率）
            # 或者取加权平均：risk = 0*p_safe + 0.5*p_medium + 1.0*p_high
            # 这里简单起见，取最后一个类（high）的概率作为风险分数
            risk_logits = p[:, -1]  # shape (H*W,)

            # 转换为 numpy
            risk_mean_flat = risk_logits.numpy()
            uncertainty_flat = u.numpy()

        # 重新 reshape 回 (H, W)
        risk_mean = risk_mean_flat.reshape(H, W)
        uncertainty = uncertainty_flat.reshape(H, W)

        return EDLGridOutput(risk_mean=risk_mean, uncertainty=uncertainty)

    except Exception as e:
        print(f"[EDL][torch] failed with error: {type(e).__name__}: {e}")
        print("[EDL][torch] falling back to placeholder output")
        # 返回占位符输出，让上层可以平滑回退
        risk_mean = np.zeros((H, W), dtype=float)
        uncertainty = np.ones((H, W), dtype=float)
        return EDLGridOutput(risk_mean=risk_mean, uncertainty=uncertainty)

