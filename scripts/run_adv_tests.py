# -*- coding: utf-8 -*-
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, '.')
from ArcticRoute.core import planner_service as ps  # type: ignore


def data_presence(ym: str):
    r = Path('ArcticRoute') / 'data_processed' / 'risk'
    has_fused = (r / f'risk_fused_{ym}.nc').exists() or any((r / f'risk_fused_{ym}_{m}.nc').exists() for m in ['linear','unetformer','poe','evidential'])
    has_escort = (r / f'R_ice_eff_{ym}.nc').exists() or (r / f'risk_ice_eff_{ym}.nc').exists()
    has_inter = (r / f'R_interact_{ym}.nc').exists() or (r / f'risk_interact_{ym}.nc').exists()
    return {'has_fused': has_fused, 'has_escort': has_escort, 'has_interact': has_inter}


def run_once(ym: str, **kw):
    env = ps.load_environment(ym, w_ice=0.7, w_accident=0.2, prior_weight=0.0, **kw)
    cost = env.cost_da
    if cost is None:
        return {'error': 'no cost_da'}
    H, W = cost.shape[-2], cost.shape[-1]
    start = (H//2, max(0, int(W*0.15)))
    goal = (H//2, min(W-1, int(W*0.85)))
    rr = ps.compute_route(env, start, goal, allow_diagonal=True, heuristic='euclidean')
    summ = ps.summarize_route(rr)
    attrs = dict(getattr(env.cost_da, 'attrs', {}) or {})
    cb = {}
    try:
        cb = ps.analyze_route_cost(env, rr)
    except Exception:
        cb = {}
    return {
        'summary': summ,
        'attrs': {k: attrs.get(k) for k in ['fusion_mode_effective','risk_agg_mode','risk_agg_mode_effective','risk_agg_alpha','use_escort','w_interact']},
        'cost_breakdown': cb,
        'escort_applied': bool(getattr(env, 'escort_applied', False)),
    }


def main():
    ym = os.environ.get('AR_YM', '202412')
    res = {'ym': ym, 'presence': data_presence(ym)}

    # baseline
    res['baseline'] = run_once(ym, fusion_mode='baseline', w_interact=0.0, use_escort=False, risk_agg_mode='mean', risk_agg_alpha=0.9)

    # fusion
    if res['presence']['has_fused']:
        res['fusion_unetformer'] = run_once(ym, fusion_mode='unetformer', w_interact=0.0, use_escort=False, risk_agg_mode='mean', risk_agg_alpha=0.9)
        # robust agg cvar
        res['fusion_unetformer_cvar'] = run_once(ym, fusion_mode='unetformer', w_interact=0.0, use_escort=False, risk_agg_mode='cvar', risk_agg_alpha=0.9)
    else:
        res['fusion_unetformer'] = {'skipped': True, 'reason': 'no fused risk file'}

    # escort
    if res['presence']['has_escort']:
        res['escort'] = run_once(ym, fusion_mode='baseline', w_interact=0.0, use_escort=True, risk_agg_mode='mean', risk_agg_alpha=0.9)
    else:
        res['escort'] = {'skipped': True, 'reason': 'no escort risk file'}

    # interaction: 若该 ym 无，则尝试其它有交互层的 ym
    if res['presence']['has_interact']:
        res['interact_0_8'] = run_once(ym, fusion_mode='baseline', w_interact=0.8, use_escort=False, risk_agg_mode='mean', risk_agg_alpha=0.9)
    else:
        # 探测一个备选 ym
        alt = None
        r = Path('ArcticRoute') / 'data_processed' / 'risk'
        for p in sorted(r.glob('R_interact_*.nc')):
            alt = p.stem.split('_')[-1]
            break
        if alt:
            res['interact_alt'] = {'ym': alt, 'presence': data_presence(alt)}
            res['interact_alt']['interact_0_8'] = run_once(alt, fusion_mode='baseline', w_interact=0.8, use_escort=False, risk_agg_mode='mean', risk_agg_alpha=0.9)
        else:
            res['interact_0_8'] = {'skipped': True, 'reason': 'no interact risk file for any ym'}

    # deltas
    try:
        b = res['baseline']['summary']
        if isinstance(res.get('fusion_unetformer'), dict) and not res['fusion_unetformer'].get('skipped'):
            f = res['fusion_unetformer']['summary']
            res['delta_baseline_vs_fusion'] = {
                'distance_km': round(float(f['distance_km']) - float(b['distance_km']), 3),
                'risk_score': round(float(f['risk_score']) - float(b['risk_score']), 5),
            }
        if isinstance(res.get('escort'), dict) and not res['escort'].get('skipped'):
            e = res['escort']['summary']
            res['delta_baseline_vs_escort'] = {
                'distance_km': round(float(e['distance_km']) - float(b['distance_km']), 3),
                'risk_score': round(float(e['risk_score']) - float(b['risk_score']), 5),
            }
        key_int = 'interact_0_8'
        if key_int in res and isinstance(res.get(key_int), dict) and not res[key_int].get('skipped') and 'summary' in res[key_int]:
            i = res[key_int]['summary']
            res['delta_baseline_vs_interact'] = {
                'distance_km': round(float(i['distance_km']) - float(b['distance_km']), 3),
                'risk_score': round(float(i['risk_score']) - float(b['risk_score']), 5),
            }
    except Exception:
        pass

    out_path = Path('outputs') / 'test_adv_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(out_path))


if __name__ == '__main__':
    main()
