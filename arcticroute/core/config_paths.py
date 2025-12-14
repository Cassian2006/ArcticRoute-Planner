"""
统一的数据路径配置模块。

提供数据根目录和子目录的路径查询接口，支持环境变量覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path


def get_data_root() -> Path:
    """
    返回数据根目录（真实网格/landmask 所在位置）。

    优先读环境变量 ARCTICROUTE_DATA_ROOT，
    否则默认使用项目根目录旁边的 ArcticRoute_data_backup。

    Returns:
        Path: 数据根目录的绝对路径

    Examples:
        >>> root = get_data_root()
        >>> root.is_absolute()
        True
    """
    env = os.getenv("ARCTICROUTE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # 默认：项目根目录的兄弟目录 ArcticRoute_data_backup
    # 例如 C:/Users/.../AR_final 和 C:/Users/.../ArcticRoute_data_backup
    here = Path(__file__).resolve()
    # arcticroute/core/config_paths.py -> arcticroute/core -> arcticroute -> 项目根
    root = here.parents[2]
    return (root.parent / "ArcticRoute_data_backup").resolve()


def get_newenv_path() -> Path:
    """
    返回 newenv 子目录路径，用于存放处理后的环境数据。

    例如 land_mask_gebco.nc、env_clean.nc 等文件所在位置。

    Returns:
        Path: data_processed/newenv 子目录的绝对路径

    Examples:
        >>> newenv = get_newenv_path()
        >>> newenv.name
        'newenv'
    """
    return get_data_root() / "data_processed" / "newenv"


