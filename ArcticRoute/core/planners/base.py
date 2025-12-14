#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一规划器后端接口 (Phase 5A)

定义 RoutePlannerBackend Protocol，支持多种规划引擎（A* / PolarRoute 等）的可切换。

接口：
    plan(start_latlon, end_latlon, **kwargs) -> list[(lat, lon)]
"""

from typing import Protocol, Tuple, List, Any


class RoutePlannerBackend(Protocol):
    """
    路由规划器后端协议。
    
    任何实现此协议的类都可以作为规划引擎使用。
    """
    
    def plan(
        self,
        start_latlon: Tuple[float, float],
        end_latlon: Tuple[float, float],
        **kwargs: Any
    ) -> List[Tuple[float, float]]:
        """
        规划一条路线。
        
        Args:
            start_latlon: (latitude, longitude) 起点
            end_latlon: (latitude, longitude) 终点
            **kwargs: 后端特定的参数
        
        Returns:
            [(lat, lon), ...] 路径点列表
        
        Raises:
            PlannerBackendError: 规划失败时抛出
        """
        ...


class PlannerBackendError(Exception):
    """规划器后端错误。"""
    pass

