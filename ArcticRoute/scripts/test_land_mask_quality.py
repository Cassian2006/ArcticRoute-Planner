# -*- coding: utf-8 -*-
"""
Land Mask 质量测试（Phase LM-1）

运行：
  python -m ArcticRoute.scripts.test_land_mask_quality
或
  python ArcticRoute/scripts/test_land_mask_quality.py

测试项：
(a) 人工标注点检查：在 land_mask_gebco.nc 网格上查询若干已知陆地/海洋点是否符合 1/0 约定；
(b) 路线 land_hits 约束：对若干典型起止点规划路线，统计路径落在陆地格的数量，要求为 0。

退出码：
  0  所有检查通过
  非0 任一检查失败
"""
from __future__ import annotations

import sys
from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from ArcticRoute.core import planner_service as ps
except Exception as e:  # pragma: no cover
    print("[FATAL] 无法导入 planner_service:", e)
    sys.exit(2)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _open_land_mask_da() -> xr.DataArray:
    base = get_project_root() / "ArcticRoute" / "data_processed" / "newenv"
    p = base / "land_mask_gebco.nc"
    if not p.exists():
        raise FileNotFoundError(f"缺失 land_mask_gebco.nc: {p}")
    ds = xr.open_dataset(p)
    try:
        var = None
        for name in ds.data_vars:
            name_l = name.lower()
            if ("land" in name_l or "mask" in name_l) and ds[name].ndim >= 2:
                var = name
                break
        if var is None:
            # 退化：唯一变量
            if len(ds.data_vars) == 1:
                var = list(ds.data_vars)[0]
            else:
                var = next(iter(ds.data_vars))
        da = ds[var]
        da.load()
        return da
    finally:
        ds.close()


def _latlon_to_ij_on_da(da: xr.DataArray, lat: float, lon: float) -> Tuple[int, int]:
    latn = next((n for n in da.coords if "lat" in n.lower()), None) or "lat"
    lonn = next((n for n in da.coords if "lon" in n.lower()), None) or "lon"
    latv = np.asarray(da[latn].values, dtype=float)
    lonv = np.asarray(da[lonn].values, dtype=float)
    # 经度假设已在 0..360 或 -180..180 的单调轴；将输入映射到相同域
    try:
        lo_min, lo_max = float(np.nanmin(lonv)), float(np.nanmax(lonv))
    except Exception:
        lo_min, lo_max = 0.0, 360.0
    lv = float(lon)
    if (lo_max > 180.0 + 1e-3) or (lo_min >= 0.0):
        # 目标轴为 0..360
        lv = lv % 360.0
        if lv < 0:
            lv += 360.0
    else:
        # 目标轴为 -180..180
        lv = ((lv + 180.0) % 360.0) - 180.0
    # 最近邻索引
    i = int(np.nanargmin(np.abs(latv - float(lat))))
    # 若经度轴跨度接近全球，使用圆周差
    span = float(lo_max - lo_min)
    if span > 300.0:
        d = np.abs(lonv - lv)
        d = np.minimum(d, 360.0 - d)
        j = int(np.nanargmin(d))
    else:
        j = int(np.nanargmin(np.abs(lonv - lv)))
    return int(np.clip(i, 0, len(latv) - 1)), int(np.clip(j, 0, len(lonv) - 1))


def test_annotated_points() -> bool:
    da = _open_land_mask_da()
    latn = next((n for n in da.coords if "lat" in n.lower()), None) or "lat"
    lonn = next((n for n in da.coords if "lon" in n.lower()), None) or "lon"
    latmin, latmax = float(da[latn].min()), float(da[latn].max())
    lonmin, lonmax = float(da[lonn].min()), float(da[lonn].max())
    print(f"[GRID] land_mask_gebco grid: lat={latmin:.3f}..{latmax:.3f} lon={lonmin:.3f}..{lonmax:.3f}")

    # 标注：选择当前裁剪范围内的典型陆地/海洋点（0..160E, 65..80N 区间）
    land_points = [
        (73.3, 56.0, "Novaya Zemlya"),
        (74.5, 100.0, "Taymyr Peninsula"),
        (71.5, 70.0, "Yamal Peninsula"),
    ]
    ocean_points = [
        (72.5, 40.0, "Barents Sea"),
        (75.0, 80.0, "Kara Sea"),
        (75.0, 120.0, "Laptev Sea"),
    ]

    ok = True
    # 陆地应为 1
    for (la, lo, name) in land_points:
        if not (latmin <= la <= latmax and lonmin <= ((lo - lonmin) % 360 + lonmin) <= lonmax):
            print(f"[SKIP] 陆地点不在网格范围内: {name} ({la},{lo})")
            continue
        i, j = _latlon_to_ij_on_da(da, la, lo)
        v = float(da.values[i, j])
        if not (v >= 0.5):
            print(f"[FAIL] 陆地点应为1: {name} ({la},{lo}) -> ij=({i},{j}) value={v}")
            ok = False
        else:
            print(f"[OK] 陆地点: {name} -> 1  (ij={i},{j})")

    # 海洋应为 0
    for (la, lo, name) in ocean_points:
        if not (latmin <= la <= latmax and lonmin <= ((lo - lonmin) % 360 + lonmin) <= lonmax):
            print(f"[SKIP] 海点不在网格范围内: {name} ({la},{lo})")
            continue
        i, j = _latlon_to_ij_on_da(da, la, lo)
        v = float(da.values[i, j])
        if not (v < 0.5):
            print(f"[FAIL] 海点应为0: {name} ({la},{lo}) -> ij=({i},{j}) value={v}")
            ok = False
        else:
            print(f"[OK] 海点: {name} -> 0  (ij={i},{j})")

    return ok


def test_routes_land_hits() -> bool:
    TEST_CASES = [
        {"name": "case1_barents_to_chukchi", "ym": "202412", "start": (69.0, 33.0), "end": (70.5, 150.0), "profile": "balanced"},
        {"name": "case2_kara_short", "ym": "202412", "start": (72.0, 60.0), "end": (73.0, 120.0), "profile": "balanced"},
        {"name": "case3_barents_to_kara_long", "ym": "202412", "start": (69.0, 33.0), "end": (73.0, 120.0), "profile": "balanced"},
    ]
    all_ok = True
    for case in TEST_CASES:
        env = ps.load_environment(ym=case["ym"], profile_name=case.get("profile", "balanced"))
        route = ps.compute_route_strict_from_latlon(
            env_ctx=env,
            start_lat=float(case["start"][0]),
            start_lon=float(case["start"][1]),
            end_lat=float(case["end"][0]),
            end_lon=float(case["end"][1]),
            allow_diagonal=True,
            heuristic="euclidean",
        )
        land_mask = getattr(env, 'land_mask', None)
        path_ij = list(getattr(route, 'path_ij', []) or [])
        land_hits = 0
        hits_ll: List[tuple] = []
        if isinstance(land_mask, np.ndarray) and path_ij:
            H, W = land_mask.shape[-2], land_mask.shape[-1]
            for (ii, jj) in path_ij:
                if 0 <= ii < H and 0 <= jj < W and land_mask[ii, jj] != 0:
                    land_hits += 1
                    if len(hits_ll) < 10:
                        try:
                            arr = ps.path_ij_to_lonlat(env, [(int(ii), int(jj))])
                            if arr and len(arr[0]) == 2:
                                hits_ll.append((int(ii), int(jj), float(arr[0][0]), float(arr[0][1])))
                        except Exception:
                            pass
        print(f"[LAND] {case['name']}: land_hits={land_hits}")
        if land_hits > 0:
            print(f"[FAIL] {case['name']}: route touches land; first_hits={hits_ll}")
            all_ok = False
    return all_ok


def main() -> None:
    print("===== LM-1 | (a) 人工标注点检查 =====")
    ok_a = test_annotated_points()
    print("===== LM-1 | (b) 路线 land_hits 约束 =====")
    ok_b = test_routes_land_hits()
    all_ok = bool(ok_a and ok_b)
    print("===== LM-1 | 结果 =====", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

