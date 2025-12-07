import os
import time
import pandas as pd
import xarray as xr

from ArcticRoute.core.predictors.ice_batch import forecast_block


def main():
    path = r"C:\Users\sgddsf\Desktop\minimum\ArcticRoute\data\raw\cmems_arc\cmems_mod_arc_phy_my_topaz4_P1M_1762143591239.nc"
    if not os.path.exists(path):
        print("File not found:", path)
        return

    ds = xr.open_dataset(path)
    # 映射变量名到 sic
    if "sic" not in ds:
        for k in ["siconc","ci","ice_conc","sea_ice_concentration"]:
            if k in ds:
                ds = ds.assign(sic=ds[k])
                break
    # 规范维度名为 y/x
    ren = {}
    if "lat" in ds.dims and "y" not in ds.dims:
        ren["lat"] = "y"
    if "lon" in ds.dims and "x" not in ds.dims:
        ren["lon"] = "x"
    if ren:
        ds = ds.rename(ren)

    print("dims:", dict(ds.dims))
    Ny, Nx = int(ds.dims.get("y",0)), int(ds.dims.get("x",0))
    by, bx = 50, 50
    y0, x0 = 0, 0
    y1, x1 = min(Ny, y0+by), min(Nx, x0+bx)
    sub = ds.isel(y=slice(y0,y1), x=slice(x0,x1))
    print("sub dims:", dict(sub.dims))

    save_dir = r"C:\Users\sgddsf\Desktop\minimum\ArcticRoute\data_processed\ice_forecast"
    os.makedirs(save_dir, exist_ok=True)

    st = time.time()
    res = forecast_block(sub, var="sic", block=(50,50), horizon=12, min_len=60, use_lstm=True, save_dir=save_dir)
    el = time.time()-st

    print("run_seconds:", round(el,2))
    print("out dims:", dict(res.dims))
    print("vars:", list(res.data_vars))
    print("time coverage:", str(pd.to_datetime(res.time.values[0]))[:10],"->", str(pd.to_datetime(res.time.values[-1]))[:10])
    print("pred flag ratio:", float(res.pred_flag.mean().values))
    fut = res.isel(time=slice(-12,None))
    print("sic_pred stats future:", float(fut.sic_pred.min().values), float(fut.sic_pred.max().values))

if __name__ == "__main__":
    main()

