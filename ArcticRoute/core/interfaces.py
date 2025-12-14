from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import xarray as xr


@dataclass
class PredictorOutput:
    """预测器预处理后的数据载体。"""

    risk: xr.DataArray
    corridor: Optional[xr.DataArray]
    lat: np.ndarray
    lon: np.ndarray
    base_time_index: int
    accident: Optional[xr.DataArray] = None
    eco_norm: Optional[xr.DataArray] = None  # Phase M: 归一化 ECO 成本栅格（可选）
    incident_lat: Optional[np.ndarray] = None
    incident_lon: Optional[np.ndarray] = None
    incident_time: Optional[np.ndarray] = None
    accident_mode: Optional[str] = None
    accident_source: Optional[str] = None


class Predictor(ABC):
    """负责加载并预处理环境数据的组件。"""

    @abstractmethod
    def prepare(self, base_time_index: int) -> PredictorOutput:
        """以指定起始时间索引准备数据。"""


class CostProvider(ABC):
    """代价计算器。"""

    @abstractmethod
    def compute(self, risk: float, corridor: Optional[float], accident: Optional[float] = None) -> float:
        """根据风险、走廊概率及事故密度计算单点代价系数。"""


@dataclass
class RouteResult:
    lat_path: np.ndarray
    lon_path: np.ndarray
    total_cost: float
    path_idx: np.ndarray
    risk_samples: np.ndarray
    corridor_samples: Optional[np.ndarray]
    accident_samples: Optional[np.ndarray]
    time_indices: Sequence[int]
    time_change_events: Sequence[tuple[int, int]]
    time_switch_nodes: Sequence[dict]
    node_count: int
    distance_m: float


class RoutePlanner(ABC):
    """路径规划器接口。"""

    @abstractmethod
    def plan(
        self,
        predictor_output: PredictorOutput,
        cost_provider: CostProvider,
        start_latlon: tuple[float, float],
        goal_latlon: tuple[float, float],
        time_step_nodes: int = 0,
        neighbor8: bool = True,
    ) -> RouteResult:
        """执行路径规划并返回结果。"""
