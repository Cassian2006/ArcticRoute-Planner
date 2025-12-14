# -*- coding: utf-8 -*-
"""
自动回归测试：起终点不变量
用法：
  python -m ArcticRoute.scripts.test_endpoint_invariants
或
  python ArcticRoute/scripts/test_endpoint_invariants.py
"""
from __future__ import annotations

import sys
import math
from typing import Tuple, Dict, Any
from collections import deque
import numpy as np

try:
    from ArcticRoute.core import planner_service as ps
except Exception as e:
    print("[FATAL] 无法导入 planner_service:", e)
    sys.exit(2)

# 简单版 haversine（km）
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

TEST_CASES = [
    {
        "name": "case1_barents_to_chukchi",
        "ym": "202412",
        "start": (69.0, 33.0),
        "end": (70.5, 150.0),
        "profile": "balanced",
    },
    {
        "name": "case2_kara_short",
        "ym": "202412",
        "start": (72.0, 60.0),
        "end": (73.0, 120.0),
        "profile": "balanced",
    },
    {
        "name": "case3_barents_to_kara_long",
        "ym": "202412",
        "start": (69.0, 33.0),
        "end": (73.0, 120.0),
        "profile": "balanced",
    },
]

# 阈值与常量
MAX_DIST_KM = 150.0
MAX_COST_FINITE = 1e6


def _extract_ij(env, route, dbg: dict) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    si = sj = gi = gj = None
    if isinstance(dbg, dict):
        try:
            if "start_ij" in dbg:
                si, sj = int(dbg["start_ij"][0]), int(dbg["start_ij"][1])
            if "end_ij" in dbg:
                gi, gj = int(dbg["end_ij"][0]), int(dbg["end_ij"][1])
        except Exception:
            si = sj = gi = gj = None
    if (si is None or sj is None) and getattr(route, "path_ij", None):
        try:
            si, sj = int(route.path_ij[0][0]), int(route.path_ij[0][1])
        except Exception:
            si = sj = None
    if (gi is None or gj is None) and getattr(route, "path_ij", None):
        try:
            gi, gj = int(route.path_ij[-1][0]), int(route.path_ij[-1][1])
        except Exception:
            gi = gj = None
    s_ij = (si, sj) if (si is not None and sj is not None) else None
    g_ij = (gi, gj) if (gi is not None and gj is not None) else None
    return s_ij, g_ij


def _ij_to_lonlat(env, ij: tuple[int, int]) -> tuple[float, float]:
    try:
        arr = ps.path_ij_to_lonlat(env, [ij])
        if arr and len(arr[0]) == 2:
            return float(arr[0][0]), float(arr[0][1])
    except Exception:
        pass
    return float('nan'), float('nan')


def run_case(case: Dict[str, Any]) -> bool:
    ym = case["ym"]
    (s_lat, s_lon) = case["start"]
    (g_lat, g_lon) = case["end"]
    profile = case.get("profile", "balanced")

    env = ps.load_environment(ym=ym, profile_name=profile)

    # 预先计算最近栅格与海上格（便于调试）
    try:
        i0_s, j0_s = ps.find_nearest_grid_index(float(s_lat), float(s_lon), env)
        i0_g, j0_g = ps.find_nearest_grid_index(float(g_lat), float(g_lon), env)
    except Exception:
        i0_s = j0_s = i0_g = j0_g = None
    try:
        si, sj, s_info = ps.find_nearest_ocean_cell(float(s_lat), float(s_lon), env, max_radius=20)
        gi, gj, g_info = ps.find_nearest_ocean_cell(float(g_lat), float(g_lon), env, max_radius=20)
    except Exception as e:
        si = sj = gi = gj = None
        s_info = {"err": str(e)}
        g_info = {"err": str(e)}

    # 严格通道：直接用经纬度规划
    route = ps.compute_route_strict_from_latlon(
        env_ctx=env,
        start_lat=float(s_lat),
        start_lon=float(s_lon),
        end_lat=float(g_lat),
        end_lon=float(g_lon),
        allow_diagonal=True,
        heuristic="euclidean",
    )

    dbg = getattr(route, 'debug', {}) or {}

    # Land-hit 约束：统计路径上踩陆的格子数量
    land_mask = getattr(env, 'land_mask', None)
    path_ij = list(getattr(route, 'path_ij', []) or [])
    land_hits = 0
    if isinstance(land_mask, np.ndarray) and path_ij:
        H_lm, W_lm = land_mask.shape[-2], land_mask.shape[-1]
        for (ii, jj) in path_ij:
            if 0 <= ii < H_lm and 0 <= jj < W_lm and land_mask[ii, jj] != 0:
                land_hits += 1
    print(f"[LAND] {case['name']}: land_hits={land_hits}")
    if land_hits > 0:
        # 打印前若干个踩陆点 (i,j,lat,lon)
        hits = []
        cnt = 0
        for (ii, jj) in path_ij:
            if isinstance(land_mask, np.ndarray):
                if 0 <= ii < land_mask.shape[0] and 0 <= jj < land_mask.shape[1] and land_mask[ii, jj] != 0:
                    la, lo = _ij_to_lonlat(env, (ii, jj))
                    hits.append((int(ii), int(jj), float(la), float(lo)))
                    cnt += 1
                    if cnt >= 10:
                        break
        print(f"[FAIL] {case['name']}: route touches land; land_hits={land_hits}; first_hits={hits}; dbg={dbg}")
        return False

    # 路径首尾点（若有）
    start_ll = route.path_lonlat[0] if getattr(route, 'path_lonlat', None) else (float('nan'), float('nan'))
    end_ll = route.path_lonlat[-1] if getattr(route, 'path_lonlat', None) else (float('nan'), float('nan'))

    # Invariant 1: 起终点距离
    d_start = haversine_km(s_lat, s_lon, float(start_ll[0]), float(start_ll[1])) if route.path_lonlat else float('inf')
    d_end = haversine_km(g_lat, g_lon, float(end_ll[0]), float(end_ll[1])) if route.path_lonlat else float('inf')

    # 补充调试信息
    s_ij, g_ij = _extract_ij(env, route, dbg)
    s0_ll = _ij_to_lonlat(env, (i0_s, j0_s)) if i0_s is not None else (float('nan'), float('nan'))
    g0_ll = _ij_to_lonlat(env, (i0_g, j0_g)) if i0_g is not None else (float('nan'), float('nan'))
    ss_ll = _ij_to_lonlat(env, (si, sj)) if si is not None else (float('nan'), float('nan'))
    gg_ll = _ij_to_lonlat(env, (gi, gj)) if gi is not None else (float('nan'), float('nan'))

    def _fmt(p):
        return f"({p[0]:.3f},{p[1]:.3f})" if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, (int,float)) for x in p) else str(p)

    # 诊断辅助：可通行性与连通性
    land_mask = env.land_mask
    cost = env.cost_da.values if env.cost_da is not None else None

    def ok_walkable(i: int, j: int) -> bool:
        if cost is None:
            return False
        H, W = cost.shape[-2], cost.shape[-1]
        if not (0 <= i < H and 0 <= j < W):
            return False
        lm_ok = True
        if land_mask is not None:
            try:
                lm = land_mask
                if lm.shape != (H, W):
                    yi = (np.linspace(0, lm.shape[-2]-1, H)).astype(int)
                    xj = (np.linspace(0, lm.shape[-1]-1, W)).astype(int)
                    lm = lm[yi[:, None], xj[None, :]]
                lm_ok = (lm[i, j] == 0)
            except Exception:
                lm_ok = True
        v = float(cost[i, j])
        return lm_ok and np.isfinite(v) and (v < 1e9)

    def connectivity_same_component(s_ij: Tuple[int,int], g_ij: Tuple[int,int]) -> Tuple[bool, int]:
        if cost is None or s_ij is None or g_ij is None:
            return False, 0
        H, W = cost.shape[-2], cost.shape[-1]
        si, sj = s_ij; gi, gj = g_ij
        if not (0 <= si < H and 0 <= sj < W and 0 <= gi < H and 0 <= gj < W):
            return False, 0
        if not ok_walkable(si, sj) or not ok_walkable(gi, gj):
            return False, 0
        q = deque(); q.append((si, sj))
        seen = np.zeros((H, W), dtype=bool)
        seen[si, sj] = True
        reached = False
        cnt = 0
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while q:
            i, j = q.popleft(); cnt += 1
            if i == gi and j == gj:
                reached = True; break
            for di, dj in nbrs:
                ii, jj = i+di, j+dj
                if 0 <= ii < H and 0 <= jj < W and not seen[ii, jj] and ok_walkable(ii, jj):
                    seen[ii, jj] = True; q.append((ii, jj))
        return reached, cnt

    def exists_valid_within_km(lat: float, lon: float, km: float = 150.0, max_r: int = 60) -> Tuple[bool, float]:
        try:
            i0, j0 = ps.find_nearest_grid_index(float(lat), float(lon), env)
        except Exception:
            return False, float('inf')
        H, W = cost.shape[-2], cost.shape[-1]
        best_km = float('inf'); found = False
        for r in range(0, int(max_r)+1):
            imin, imax = max(0, i0-r), min(H-1, i0+r)
            jmin, jmax = max(0, j0-r), min(W-1, j0+r)
            # 扫描四条边，减少重复
            for ii in range(imin, imax+1):
                for jj in (jmin, jmax):
                    if ok_walkable(ii, jj):
                        la, lo = _ij_to_lonlat(env, (ii, jj))
                        d = haversine_km(lat, lon, la, lo)
                        if d < best_km:
                            best_km = d
                            if d <= km:
                                return True, d
            for jj in range(jmin+1, jmax):
                for ii in (imin, imax):
                    if ok_walkable(ii, jj):
                        la, lo = _ij_to_lonlat(env, (ii, jj))
                        d = haversine_km(lat, lon, la, lo)
                        if d < best_km:
                            best_km = d
                            if d <= km:
                                return True, d
        return found, best_km

    # 若不可达，打印连通性诊断
    if not route.reachable or not getattr(route, 'path_lonlat', None):
        same_comp, explored = connectivity_same_component((si, sj), (gi, gj)) if si is not None and gi is not None else (False, 0)
        has_near_start, min_km_start = exists_valid_within_km(s_lat, s_lon, 150.0)
        has_near_end, min_km_end = exists_valid_within_km(g_lat, g_lon, 150.0)
        print(f"[FAIL] {case['name']}: route not reachable; s_in=({_fmt((s_lat,s_lon))}), g_in=({_fmt((g_lat,g_lon))}); grid_i0j0={(i0_s,j0_s)}->{_fmt(s0_ll)} / {(i0_g,j0_g)}->{_fmt(g0_ll)}; ocean_snap={(si,sj)}->{_fmt(ss_ll)} / {(gi,gj)}->{_fmt(gg_ll)}; same_component={same_comp} explored={explored} near_start150={has_near_start} (min={min_km_start:.1f}km) near_end150={has_near_end} (min={min_km_end:.1f}km); dbg={dbg}")
        return False

    if d_start > MAX_DIST_KM or d_end > MAX_DIST_KM:
        same_comp, explored = connectivity_same_component((si, sj), (gi, gj)) if si is not None and gi is not None else (False, 0)
        has_near_start, min_km_start = exists_valid_within_km(s_lat, s_lon, 150.0)
        has_near_end, min_km_end = exists_valid_within_km(g_lat, g_lon, 150.0)
        print(f"[FAIL] {case['name']}: dist_start={d_start:.1f}, dist_end={d_end:.1f}; s_in=({_fmt((s_lat,s_lon))}), g_in=({_fmt((g_lat,g_lon))}); start_ll={_fmt(start_ll)}, end_ll={_fmt(end_ll)}; grid_i0j0={(i0_s,j0_s)}->{_fmt(s0_ll)} / {(i0_g,j0_g)}->{_fmt(g0_ll)}; ocean_snap={(si,sj)}->{_fmt(ss_ll)} / {(gi,gj)}->{_fmt(gg_ll)}; s_ij={s_ij}, g_ij={g_ij}; same_component={same_comp} explored={explored} near_start150={has_near_start} (min={min_km_start:.1f}km) near_end150={has_near_end} (min={min_km_end:.1f}km); dbg={dbg}")
        return False

    # Invariant 2: 海上 + cost 有限（更严格 cost<1e6 作为 UI 稳定性要求）
    def ok_ij(i: int, j: int) -> bool:
        if cost is None:
            return False
        H, W = cost.shape[-2], cost.shape[-1]
        if not (0 <= i < H and 0 <= j < W):
            return False
        if isinstance(land_mask, type(None)):
            lm_ok = True
        else:
            try:
                lm = land_mask
                if lm.shape != (H, W):
                    yi = (np.linspace(0, lm.shape[-2]-1, H)).astype(int)
                    xj = (np.linspace(0, lm.shape[-1]-1, W)).astype(int)
                    lm = lm[yi[:, None], xj[None, :]]
                lm_ok = (lm[i, j] == 0)
            except Exception:
                lm_ok = True
        v = float(cost[i, j])
        return lm_ok and math.isfinite(v) and v < 1e6

    if not ok_ij(*s_ij) or not ok_ij(*g_ij):
        print(f"[FAIL] {case['name']}: start/goal not valid ocean cell; s_ij={s_ij}, g_ij={g_ij}; s_ll={_fmt(_ij_to_lonlat(env, s_ij)) if s_ij else 'NA'}, g_ll={_fmt(_ij_to_lonlat(env, g_ij)) if g_ij else 'NA'}; dbg={dbg}")
        return False

    steps = getattr(route, 'len', 0) or len(getattr(route, 'path_ij', []) or [])
    print(f"[OK] {case['name']}: land_hits={land_hits}, d_start={d_start:.1f}, d_end={d_end:.1f}, steps={int(steps)}")
    return True


def main() -> None:
    all_ok = True
    for case in TEST_CASES:
        if not run_case(case):
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
