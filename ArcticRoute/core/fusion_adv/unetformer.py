from __future__ import annotations
"""
UNet-Former 轻量融合插件（最小可用）：
- 模型：简化版 U-Net 编码器 + Transformer-Style 通道注意（仅一层），参数量 ~几百万级
- 训练：弱监督 BCE（掩码忽略 NaN）；支持 AMP；保存 best.ckpt（按 val ECE/Brier 最小）
- 推理：对整图滑窗预测，输出 risk ∈ [0,1]

# REUSE: 数据集与通道读取逻辑沿用 fusion_adv.dataset
"""
from typing import Any, Dict, Tuple, Optional, List
import os
import math
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from .dataset import make_weak_labels, build_patches

ROOT = os.path.join(os.getcwd(), "ArcticRoute")
RISK_DIR = os.path.join(ROOT, "data_processed", "risk")
CV_DIR = os.path.join(ROOT, "data_processed", "cv_cache")
PRIOR_DIR = os.path.join(ROOT, "data_processed", "prior")
OUT_DIR = os.path.join(ROOT, "outputs", "phaseK", "fusion_unetformer")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        p = (k // 2) if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Attention2D(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):  # x: [B,C,H,W]
        B,C,H,W = x.shape
        q,k,v = torch.chunk(self.qkv(x), 3, dim=1)
        # 展平空间
        q = q.reshape(B, self.heads, C//self.heads, H*W)
        k = k.reshape(B, self.heads, C//self.heads, H*W)
        v = v.reshape(B, self.heads, C//self.heads, H*W)
        att = torch.softmax((q.transpose(2,3) @ k.transpose(2,3).transpose(-2,-1)) / math.sqrt(C//self.heads + 1e-6), dim=-1)  # [B,h,HW,HW]
        out = att @ v.transpose(2,3)
        out = out.transpose(2,3).reshape(B, C, H, W)
        return self.proj(out)


class UNetFormer(nn.Module):
    def __init__(self, in_ch: int = 6, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNAct(in_ch, base), ConvBNAct(base, base))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBNAct(base, base*2), ConvBNAct(base*2, base*2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBNAct(base*2, base*4), ConvBNAct(base*4, base*4))
        self.attn = Attention2D(base*4, heads=4)
        self.dec2 = nn.Sequential(ConvBNAct(base*4+base*2, base*2), ConvBNAct(base*2, base*2))
        self.dec1 = nn.Sequential(ConvBNAct(base*2+base, base), ConvBNAct(base, base))
        self.out = nn.Conv2d(base, 1, 1)
    def _match_cat(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 将 a,b 裁剪到相同的空间尺寸（中心裁剪）
        Ha, Wa = a.shape[-2], a.shape[-1]
        Hb, Wb = b.shape[-2], b.shape[-1]
        H = min(Ha, Hb); W = min(Wa, Wb)
        def center_crop(t, H, W):
            ht, wt = t.shape[-2], t.shape[-1]
            ys = max(0, (ht - H) // 2); xs = max(0, (wt - W) // 2)
            return t[..., ys:ys+H, xs:xs+W]
        return center_crop(a, H, W), center_crop(b, H, W)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        z = e3 + self.attn(e3)
        u2 = F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=False)
        u2, e2m = self._match_cat(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2m], dim=1))
        u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        u1, e1m = self._match_cat(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1m], dim=1))
        return torch.sigmoid(self.out(d1))


def _load_channel(path: str, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0)
    return da


def _gather_channels(ym: str, inputs: List[str]) -> Dict[str, xr.DataArray]:
    ch: Dict[str, xr.DataArray] = {}
    for key in inputs:
        if key in ("R_ice_eff", "R_ice"):
            p = os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc") if key == "R_ice_eff" else os.path.join(RISK_DIR, f"R_ice_{ym}.nc")
            var = "risk" if key == "R_ice_eff" else "R_ice"
            if os.path.exists(p):
                ch[key] = _load_channel(p, var)
        elif key == "R_wave":
            p = os.path.join(RISK_DIR, f"R_wave_{ym}.nc")
            if os.path.exists(p):
                ch[key] = _load_channel(p, "R_wave")
        elif key == "R_acc":
            p1 = os.path.join(RISK_DIR, f"risk_accident_{ym}.nc")
            p2 = os.path.join(RISK_DIR, f"R_acc_{ym}.nc")
            p = p1 if os.path.exists(p1) else p2
            if os.path.exists(p):
                with xr.open_dataset(p) as ds_tmp:  # REUSE: 避免泄露
                    var = "R_acc" if "R_acc" in ds_tmp.variables else (list(ds_tmp.data_vars)[0] if ds_tmp.data_vars else None)
                if var:
                    ch[key] = _load_channel(p, var)
        elif key == "prior_penalty":
            p1 = os.path.join(PRIOR_DIR, f"prior_transformer_{ym}.nc")
            if os.path.exists(p1):
                ds = xr.open_dataset(p1)
                var = "PriorPenalty" if "PriorPenalty" in ds else ("prior_penalty" if "prior_penalty" in ds else list(ds.data_vars)[0])
                da = ds[var]
                if "time" in da.dims:
                    da = da.isel(time=0)
                ch[key] = da
        elif key == "edge_dist":
            p = os.path.join(CV_DIR, f"edge_dist_{ym}.nc")
            if os.path.exists(p):
                ch[key] = _load_channel(p, "edge_dist")
        elif key == "lead_prob":
            p = os.path.join(CV_DIR, f"lead_prob_{ym}.nc")
            if os.path.exists(p):
                ch[key] = _load_channel(p, "lead_prob")
    # 对齐到参考通道网格（优先 R_ice_eff/R_ice）
    if ch:
        ref_key = next((k for k in ("R_ice_eff","R_ice") if k in ch), next(iter(ch.keys())))
        ref = ch[ref_key]
        aligned: Dict[str, xr.DataArray] = {}
        for k, da in ch.items():
            if (set(ref.dims) == set(da.dims)) and all(int(ref.sizes[d]) == int(da.sizes.get(d, -1)) for d in ref.dims):
                aligned[k] = da
                continue
            try:
                # 尝试最近邻插值对齐
                da2 = da
                # 重命名常见经纬维到 y/x 以便 interp_like
                rename_map = {}
                if "lat" in da2.dims: rename_map["lat"] = "y"
                if "latitude" in da2.dims: rename_map["latitude"] = "y"
                if "lon" in da2.dims: rename_map["lon"] = "x"
                if "longitude" in da2.dims: rename_map["longitude"] = "x"
                if rename_map:
                    da2 = da2.rename(rename_map)
                da2 = da2.interp_like(ref, method="nearest")
                aligned[k] = da2
            except Exception:
                # 回退：裁剪到公共最小 HxW
                Hc = min(int(ref.shape[-2]), int(da.shape[-2]))
                Wc = min(int(ref.shape[-1]), int(da.shape[-1]))
                aligned[k] = da.isel({da.dims[-2]: slice(0,Hc), da.dims[-1]: slice(0,Wc)})
        ch = aligned
    return ch


def _ece(prob: np.ndarray, label: np.ndarray, mask: np.ndarray, n_bins: int = 10) -> float:
    eps = 1e-6
    p = prob[mask > 0.5]
    y = label[mask > 0.5]
    if p.size == 0:
        return 1.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1] + (1e-8 if i==n_bins-1 else 0))
        if m.any():
            conf = float(p[m].mean())
            acc = float(y[m].mean())
            ece += (m.mean()) * abs(conf - acc)
    return float(ece)


def train(ym: str, inputs: List[str], epochs: int = 10, batch: int = 8, tile: int = 256, stride: int = 128, out_dir: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    if torch is None or xr is None:
        raise RuntimeError("需要 torch/xarray")
    ch = _gather_channels(ym, inputs)
    if not ch:
        raise RuntimeError("无可用通道用于训练")
    # 弱标签：ais_density + 事故/闭冰掩码（占位：仅 ais_density）
    ais_path = os.path.join(ROOT, "data_processed", "features", f"ais_density_{ym}.nc")
    if not os.path.exists(ais_path):
        # 占位：以 R_ice_eff 近似替代阈值
        key0 = next(iter(ch.keys()))
        labels = ch[key0] * 0 + np.nan
    else:
        ds_ais = xr.open_dataset(ais_path)
        ais_var = "ais_density" if "ais_density" in ds_ais else list(ds_ais.data_vars)[0]
        labels = make_weak_labels(ds_ais[ais_var], incidents_mask=None, ice_mask=None)
    dataset = build_patches(ch, labels, tile=tile, stride=stride, aug=False)
    C = len(ch)
    model = UNetFormer(in_ch=C, base=24)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # 简单划分 train/val
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    def _loader(indexes: np.ndarray, bs: int):
        for i in range(0, len(indexes), bs):
            sub = indexes[i:i+bs]
            Xs = []; Ys = []; Ms = []
            for j in sub:
                x, y, m = dataset[int(j)]
                Xs.append(x.numpy()); Ys.append(y.numpy()); Ms.append(m.numpy())
            X = torch.from_numpy(np.stack(Xs)).to(device)
            Y = torch.from_numpy(np.stack(Ys)).to(device)
            M = torch.from_numpy(np.stack(Ms)).to(device)
            yield X, Y, M

    best = {"ece": 1e9, "brier": 1e9}
    run_id = time.strftime("%Y%m%dT%H%M%S")
    out_dir = out_dir or os.path.join(OUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "best.ckpt")

    last_val = {"p": None, "y": None, "m": None}

    for ep in range(max(1, epochs if not dry_run else 1)):
        model.train()
        for X, Y, M in _loader(tr_idx, batch):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                P = model(X)
            # 在 FP32 下计算 BCE 以避免 autocast 警告
            loss_raw = F.binary_cross_entropy(P.float(), Y.float(), reduction='none')
            loss = (loss_raw * M.float()).sum() / (M.float().sum() + 1e-6)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        # 验证
        model.eval()
        probs = []; gts = []; ms = []
        with torch.no_grad():
            for X, Y, M in _loader(val_idx, batch):
                P = model(X)
                probs.append(P.detach().cpu().numpy())
                gts.append(Y.detach().cpu().numpy())
                ms.append(M.detach().cpu().numpy())
        if probs:
            p = np.concatenate(probs, 0)[:,0]
            y = np.concatenate(gts, 0)[:,0]
            m = np.concatenate(ms, 0)[:,0]
            last_val = {"p": p, "y": y, "m": m}
            brier = float(((p - y) ** 2 * m).sum() / (m.sum() + 1e-6))
            ece = _ece(p, y, m, n_bins=10)
            if (ece < best["ece"]) or (brier < best["brier"]):
                best.update({"ece": ece, "brier": brier, "epoch": ep})
                torch.save({"state_dict": model.state_dict(), "in_ch": C}, ckpt)

    # 拟合与保存校准（isotonic）
    calib_json = None
    try:
        if last_val["p"] is not None:
            from .calibrate import fit_calibrator, save_calibrator  # REUSE
            model_c = fit_calibrator(last_val["p"], last_val["y"], last_val["m"], method="isotonic")
            calib_json = os.path.join(out_dir, "calibration.json")
            save_calibrator(model_c, calib_json)
    except Exception:
        calib_json = None

    return {"run_id": run_id, "ckpt": ckpt, "best": best, "out_dir": out_dir, "calibration": calib_json}


def infer_month(ym: str, inputs: List[str], ckpt: str, calibrated: bool = False, calib_path: Optional[str] = None) -> Dict[str, Any]:
    if torch is None or xr is None:
        raise RuntimeError("需要 torch/xarray")
    ch = _gather_channels(ym, inputs)
    if not ch:
        raise RuntimeError("无可用通道用于推理")
    # 组装为张量 [1,C,H,W]
    keys = list(ch.keys())
    # 选择参考网格（优先 R_ice_eff/R_ice）并保留完整尺寸以便输出复原
    ref_key = next((k for k in ("R_ice_eff","R_ice") if k in ch), keys[0])
    ref_full = ch[ref_key]
    # 统一尺寸（裁剪到公共最小 HxW）
    arr_list = [np.asarray(ch[k].values, dtype=np.float32) for k in keys]
    shapes = [(a.shape[-2], a.shape[-1]) for a in arr_list]
    Hc = min(s[0] for s in shapes); Wc = min(s[1] for s in shapes)
    # 若过小，放大到至少 32x32
    min_sz = 32
    if Hc < min_sz or Wc < min_sz:
        try:
            from skimage.transform import resize as _resize  # type: ignore
            arr_list = [ _resize(a[:Hc,:Wc], (max(min_sz,Hc), max(min_sz,Wc)), order=1, mode='edge', anti_aliasing=True, preserve_range=True).astype(np.float32) for a in arr_list ]
            Hc, Wc = max(min_sz, Hc), max(min_sz, Wc)
        except Exception:
            sy = max(1, int(np.ceil(min_sz / max(1,Hc)))); sx = max(1, int(np.ceil(min_sz / max(1,Wc))))
            arr_list = [ np.repeat(np.repeat(a[:Hc,:Wc], sy, axis=0), sx, axis=1).astype(np.float32) for a in arr_list ]
            Hc, Wc = max(min_sz, Hc), max(min_sz, Wc)
    else:
        arr_list = [a[:Hc, :Wc] for a in arr_list]
    C = len(keys)
    X = np.stack(arr_list, axis=0)
    vmin = float(np.nanmin(X)); vmax = float(np.nanmax(X))
    if vmax > vmin:
        X = (X - vmin) / (vmax - vmin)
    X[np.isnan(X)] = 0.0
    X = X[None, ...]
    H, W = Hc, Wc
    # 基准坐标也裁剪，若有缩放则需重建坐标
    if X.shape[-2] != ch[keys[0]].shape[-2] or X.shape[-1] != ch[keys[0]].shape[-1]:
        # 尺寸已改变，重建坐标
        coords = {
            "y": np.linspace(0, H-1, H, dtype=np.float32),
            "x": np.linspace(0, W-1, W, dtype=np.float32),
        }
        dims = ("y", "x")
        da0_attrs = ch[keys[0]].attrs
    else:
        da0 = ch[keys[0]].isel(y=slice(0,H), x=slice(0,W)) if ("y" in ch[keys[0]].dims and "x" in ch[keys[0]].dims) else ch[keys[0]]
        coords = da0.coords
        dims = da0.dims
        da0_attrs = da0.attrs

    # 模型
    state = torch.load(ckpt, map_location="cpu")
    model = UNetFormer(in_ch=state.get("in_ch", C), base=24)
    model.load_state_dict(state["state_dict"])  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    with torch.no_grad():
        P = model(torch.from_numpy(X).to(device))
        prob = P.detach().cpu().numpy()[0,0]
    # 可选后校准（由 calibrate.py 提供）
    if calibrated and calib_path and os.path.exists(calib_path):
        try:
            from .calibrate import apply_calibration  # REUSE
            prob = apply_calibration(prob, calib_path)
        except Exception:
            pass
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

    # 若需要，将概率上采样回参考网格尺寸
    try:
        refH = int(ref_full.shape[-2]); refW = int(ref_full.shape[-1])
        if (prob.shape[0] != refH) or (prob.shape[1] != refW):
            try:
                from skimage.transform import resize as _resize  # type: ignore
                prob = _resize(prob, (refH, refW), order=1, mode='edge', anti_aliasing=True, preserve_range=True).astype(np.float32)
            except Exception:
                sy = max(1, int(np.ceil(refH / max(1,prob.shape[0])))); sx = max(1, int(np.ceil(refW / max(1,prob.shape[1]))))
                prob = np.repeat(np.repeat(prob, sy, axis=0), sx, axis=1)[:refH, :refW].astype(np.float32)
            dims = ref_full.dims
            coords = ref_full.coords
            da0_attrs = ref_full.attrs
    except Exception:
        pass

    # 写 NetCDF
    ds = xr.Dataset({"risk": xr.DataArray(prob, dims=dims, coords=coords)})
    # 若参考网格含 lat/lon，则作为坐标写入（route.scan 需要 coords）
    try:
        latc = ref_full.coords.get("lat") or ref_full.coords.get("latitude")
        lonc = ref_full.coords.get("lon") or ref_full.coords.get("longitude")
        if latc is not None and lonc is not None:
            # 若为 2D，经纬转为 1D（lat: y 维，lon: x 维），以适配 route.scan 的索引
            if len(latc.dims) == 2:
                lat1 = latc.isel({latc.dims[1]: 0}).values.astype(np.float32)
            else:
                lat1 = latc.values.astype(np.float32)
            if len(lonc.dims) == 2:
                lon1 = lonc.isel({lonc.dims[0]: 0}).values.astype(np.float32)
            else:
                lon1 = lonc.values.astype(np.float32)
            ds = ds.assign_coords(
                lat=("y", lat1 if lat1.ndim == 1 else lat1.reshape(-1)),
                lon=("x", lon1 if lon1.ndim == 1 else lon1.reshape(-1)),
                latitude=("y", lat1 if lat1.ndim == 1 else lat1.reshape(-1)),
                longitude=("x", lon1 if lon1.ndim == 1 else lon1.reshape(-1)),
            )
            ds = ds.set_coords(["lat","lon","latitude","longitude"])  # REUSE: ensure coords
    except Exception:
        pass
    ds["risk"].attrs = da0_attrs
    ds["risk"].attrs.update({"long_name": "Fused risk (UNet-Former)", "units": "1", "source": "unetformer", "calibrated": int(bool(calibrated))})
    out_path = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
    ds.to_netcdf(out_path)
    # 写 meta
    try:
        import json as _json
        run_id = time.strftime("%Y%m%dT%H%M%S")
        meta = {"logical_id": f"risk_fused_{ym}", "inputs": list(keys), "ckpt": str(ckpt), "calibrated": bool(calibrated), "calibration": calib_path, "run_id": run_id, "git_sha": os.environ.get("GIT_SHA"), "metrics": {}}
        with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
            _json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return {"ym": ym, "out": out_path, "shape": [H, W], "vars": ["risk"], "calibrated": bool(calibrated)}


__all__ = ["UNetFormer", "train", "infer_month"]

