from __future__ import annotations
import json, sys
from pathlib import Path

def pick(obj, product_id: str, prefer_keywords: list[str], prefer_var_keywords: list[str]):
    """
    从 describe JSON 结果中提取 dataset_id 和变量名
    
    Args:
        obj: describe 返回的 JSON 对象
        product_id: 产品 ID（用于验证）
        prefer_keywords: 优先匹配的关键词列表（用于 dataset_id）
        prefer_var_keywords: 优先匹配的关键词列表（用于变量名）
    
    Returns:
        dict with keys: dataset_id, variables
    """
    # obj is a big search result; structure can evolve -> defensive parsing
    text = json.dumps(obj).lower()
    if product_id.lower() not in text:
        return None

    # Heuristic: find first dataset id string looking like cmems_... in the payload
    dataset_ids = set()
    def walk(x):
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, str) and k.lower() in ("dataset_id","datasetid","dataset-id","id"):
                    if "cmems_" in v.lower() or "dataset" in v.lower():
                        dataset_ids.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(obj)

    dataset_id = None
    for cand in dataset_ids:
        s = cand.lower()
        if all(k in s for k in prefer_keywords):
            dataset_id = cand
            break
    if dataset_id is None and dataset_ids:
        dataset_id = sorted(dataset_ids)[0]

    # Variables: search common fields
    vars_ = set()
    def walk_vars(x):
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, str) and k.lower() in ("variable","variables","short_name","standard_name","name"):
                    vars_.add(v)
                walk_vars(v)
        elif isinstance(x, list):
            for v in x: walk_vars(v)
    walk_vars(obj)

    chosen_vars = []
    for vk in prefer_var_keywords:
        for v in sorted(vars_):
            if vk in v.lower():
                chosen_vars.append(v)
        if chosen_vars:
            break

    return {"dataset_id": dataset_id, "variables": sorted(set(chosen_vars))}

def main():
    # 尝试从 cmems_sic_describe.json 或 cmems_swh_describe.json 读取
    sic_path = Path("reports/cmems_sic_describe.json")
    swh_path = Path("reports/cmems_swh_describe.json")
    wav_path = Path("reports/cmems_wav_describe.json")
    
    # 优先使用 sic_describe
    if not sic_path.exists():
        print(f"[ERROR] {sic_path} 不存在，请先运行 describe 命令")
        return
    
    # 波浪数据：优先 swh，其次 wav
    if swh_path.exists():
        wave_path = swh_path
        wave_product = "dataset-wam-arctic-1hr3km-be"
    elif wav_path.exists():
        wave_path = wav_path
        wave_product = "dataset-wam-arctic-1hr3km-be"
    else:
        print(f"[ERROR] {swh_path} 和 {wav_path} 都不存在，请先运行 describe 命令")
        return
    
    sic = json.loads(sic_path.read_text(encoding="utf-8-sig"))
    wave = json.loads(wave_path.read_text(encoding="utf-8-sig"))

    out = {}
    # Phase 9: 使用真实 dataset id
    out["sic"] = pick(sic, "cmems_mod_arc_phy_anfc_nextsim_hm", prefer_keywords=["nextsim","arc","phy","ice"], prefer_var_keywords=["siconc","sic","conc","sea_ice"])
    out["wav"] = pick(wave, wave_product, prefer_keywords=["wam","arctic","3km"], prefer_var_keywords=["vhm0","sea_surface_wave_significant_height","swh","hs","significant"])

    Path("reports/cmems_resolved.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[OK] 已写入 reports/cmems_resolved.json")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

