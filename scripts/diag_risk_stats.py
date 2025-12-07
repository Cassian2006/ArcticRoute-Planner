# -*- coding: utf-8 -*-
import os
import sys
import json
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from ArcticRoute.core import planner_service as ps  # type: ignore

CAND_VARS_FUSED = ["risk_fused", "risk", "prob_risk", "rv_fused"]
CAND_VARS_INTER = ["risk_interact", "R_interact", "interact", "risk"]
CAND_VARS_ESCORT = ["R_ice_eff", "risk_ice_eff", "risk_ice", "risk"]


def open_da(path: Path, var_cands):
    if not path.exists():
        return None, None
    try:
        with xr.open_dataset(path) as ds:
            v = None
            for k in var_cands:
                if k in ds.data_vars:
                    v = k
                    break
            if v is None and ds.data_vars:
                v = list(ds.data_vars)[0]
            if v is None:
                return None, None
            da = ds[v]
            if 'time' in da.dims:
                da = da.isel(time=0)
            return da.load(), v
    except Exception:
        return None, None


def da_stats(da: xr.DataArray):
    arr = np.asarray(da.values)
    mask = np.isfinite(arr)
    total = arr.size
    finite = int(np.count_nonzero(mask))
    if finite == 0:
        return {
            'shape': tuple(int(s) for s in arr.shape),
            'finite': 0,
            'total': int(total),
            'nan_ratio': None if total == 0 else float((total - 0) / total),
            'q01': None, 'q05': None, 'q50': None, 'q95': None, 'q99': None,
            'sample_dim': None,
        }
    vals = arr[mask].astype('float64').ravel()
    def q(p):
        try:
            return float(np.nanquantile(vals, p))
        except Exception:
            return None
    dims = list(da.dims)
    sample_dim = next((d for d in ('sample','member','ensemble') if d in dims), None)
    return {
        'shape': tuple(int(s) for s in arr.shape),
        'finite': int(finite),
        'total': int(total),
        'nan_ratio': float((total - finite) / total),
        'q01': q(0.01), 'q05': q(0.05), 'q50': q(0.5), 'q95': q(0.95), 'q99': q(0.99),
        'sample_dim': sample_dim,
    }


def normalize_q(da: xr.DataArray, qlo=0.05, qhi=0.95):
    arr = np.asarray(da.values, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return xr.zeros_like(da)
    vs = arr[mask]
    try:
        lo = float(np.nanquantile(vs, qlo))
        hi = float(np.nanquantile(vs, qhi))
    except Exception:
        lo, hi = float(np.nanmin(vs)), float(np.nanmax(vs))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype('float32')
    out = xr.DataArray(norm, dims=da.dims)
    return out


def main():
    ym = os.environ.get('AR_YM','202412')
    risk_dir = Path('ArcticRoute')/'data_processed'/'risk'
    out = {'ym': ym}

    # Load baseline env for grid reference and route sampling
    env = ps.load_environment(ym, w_ice=0.7, w_accident=0.2, prior_weight=0.0,
                              fusion_mode='baseline', w_interact=0.0, use_escort=False,
                              risk_agg_mode='mean', risk_agg_alpha=0.9)
    cost = env.cost_da
    H, W = cost.shape[-2], cost.shape[-1]
    start = (H//2, max(0, int(W*0.15)))
    goal = (H//2, min(W-1, int(W*0.85)))
    rr = ps.compute_route(env, start, goal, True, 'euclidean')
    path = rr.path_ij

    # Fused
    fused_paths = [
        risk_dir / f"risk_fused_{ym}_unetformer.nc",
        risk_dir / f"risk_fused_{ym}.nc",
    ]
    fused_da, fused_var = None, None
    for p in fused_paths:
        fused_da, fused_var = open_da(p, CAND_VARS_FUSED)
        if fused_da is not None:
            fused_src = str(p.name)
            break
    if fused_da is None:
        out['fused'] = {'available': False}
    else:
        # aggregate if sample dim
        dims = list(fused_da.dims)
        sample_dim = next((d for d in ('sample','member','ensemble') if d in dims), None)
        fused_agg = fused_da
        if sample_dim is not None:
            try:
                fused_agg = fused_da.mean(dim=sample_dim)
                out.setdefault('notes',[]).append('fused_has_sample_dim')
            except Exception:
                pass
        # align
        try:
            if fused_agg.shape[-2:] != (H,W):
                fused_agg = fused_agg.interp_like(cost)
        except Exception:
            pass
        stats_raw = da_stats(fused_da)
        stats_agg = da_stats(fused_agg)
        fused_norm = normalize_q(fused_agg, 0.05, 0.95)
        stats_norm = da_stats(fused_norm)
        # sample along path avg
        try:
            arrN = np.asarray(fused_norm.values)
            s=0.0; c=0
            for (i,j),(ni,nj) in zip(path[:-1], path[1:]):
                if 0<=i<arrN.shape[0] and 0<=j<arrN.shape[1]:
                    s += float(arrN[i,j])
                    c += 1
            path_avg = float(s/c) if c>0 else None
        except Exception:
            path_avg = None
        out['fused'] = {
            'available': True,
            'file': fused_src,
            'var': fused_var,
            'stats_raw': stats_raw,
            'stats_agg': stats_agg,
            'stats_norm': stats_norm,
            'path_norm_avg': path_avg,
        }

    # Escort
    escort_paths = [
        risk_dir / f"R_ice_eff_{ym}.nc",
        risk_dir / f"risk_ice_eff_{ym}.nc",
    ]
    escort_da, escort_var = None, None
    for p in escort_paths:
        escort_da, escort_var = open_da(p, CAND_VARS_ESCORT)
        if escort_da is not None:
            escort_src = str(p.name)
            break
    if escort_da is None:
        out['escort'] = {'available': False}
    else:
        da2 = escort_da
        try:
            if da2.shape[-2:] != (H,W):
                da2 = da2.interp_like(cost)
        except Exception:
            pass
        stats_raw = da_stats(escort_da)
        stats_aln = da_stats(da2)
        esc_norm = normalize_q(da2, 0.05, 0.95)
        stats_norm = da_stats(esc_norm)
        # sample along path avg
        try:
            arrN = np.asarray(esc_norm.values)
            s=0.0; c=0
            for (i,j),(ni,nj) in zip(path[:-1], path[1:]):
                if 0<=i<arrN.shape[0] and 0<=j<arrN.shape[1]:
                    s += float(arrN[i,j])
                    c += 1
            path_avg = float(s/c) if c>0 else None
        except Exception:
            path_avg = None
        out['escort'] = {
            'available': True,
            'file': escort_src,
            'var': escort_var,
            'stats_raw': stats_raw,
            'stats_aligned': stats_aln,
            'stats_norm': stats_norm,
            'path_norm_avg': path_avg,
        }

    # Write
    out_path = Path('outputs')/f'diag_risk_stats_{ym}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(out_path))

if __name__ == '__main__':
    main()






















