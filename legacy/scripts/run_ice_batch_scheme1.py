import os
import time
import pandas as pd
import xarray as xr

from ArcticRoute.core.predictors.ice_batch import forecast_block

def main():
    path = r"C:\Users\sgddsf\Desktop\minimum\ArcticRoute\data\raw\cmems_arc\cmems_mod_arc_phy_my_topaz4_P1M_1762143591239.nc"
    save_dir = r"C:\Users\sgddsf\Desktop\minimum\ArcticRoute\data_processed\ice_forecast"

    if not os.path.exists(path):
        print("File not found:", path)
        return

    os.makedirs(save_dir, exist_ok=True)

    ds = xr.open_dataset(path)
    # 变量映射
    if "sic" not in ds:
        for k in ["siconc","ci","ice_conc","sea_ice_concentration"]:
            if k in ds:
                ds = ds.assign(sic=ds[k])
                break
    if "sit" not in ds:
        for k in ["sithick_corr","sithick","sea_ice_thickness","ice_thickness"]:
            if k in ds:
                ds = ds.assign(sit=ds[k])
                break
    # 维度规范
    ren = {}
    if "latitude" in ds.dims and "y" not in ds.dims:
        ren["latitude"] = "y"
    if "longitude" in ds.dims and "x" not in ds.dims:
        ren["longitude"] = "x"
    if ren:
        ds = ds.rename(ren)

    print("dims:", dict(ds.sizes))

    st = time.time()
    res = forecast_block(
        ds, var="sic",
        block=(50,50), horizon=12,
        min_len=120, min_valid_frac=0.6, std_thresh=1e-6, per_month_min=2,
        use_lstm=True,
        save_dir=save_dir,
        max_blocks=5,          # 方案一：先跑前 5 个块
        dry_run=False,
        max_workers=2          # 控制并行进程数，缓解内存
    )
    el = time.time()-st

    print("run_seconds:", round(el,2))
    print("out dims:", dict(res.sizes))
    print("vars:", list(res.data_vars))
    print("time coverage:", str(pd.to_datetime(res.time.values[0]))[:10],"->", str(pd.to_datetime(res.time.values[-1]))[:10])
    print("pred flag ratio:", float(res.pred_flag.mean().values))
    fut = res.isel(time=slice(-12,None))
    print("sic_pred stats future:", float(fut.sic_pred.min().values), float(fut.sic_pred.max().values))

if __name__ == "__main__":
    main()




















