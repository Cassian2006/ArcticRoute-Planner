import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
import geodatasets as gd


def main() -> None:
    env_path = r"ArcticRoute/data_processed/env/env_clean.nc"
    out_path = r"ArcticRoute/data_processed/env/land_mask.nc"

    ds = xr.open_dataset(env_path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # wrap lon to [-180, 180] to match Natural Earth
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    LonW, Lat = np.meshgrid(lon_wrapped, lat)

    # load land polygons via geodatasets
    try:
        land_path = gd.get_path("naturalearth.land")
        gdf = gpd.read_file(land_path)
    except Exception:
        ctry_path = gd.get_path("naturalearth.cultural_admin_0_countries")
        gdf = gpd.read_file(ctry_path)

    geom = gdf.unary_union
    poly_prepped = prep(geom)

    H, W = Lat.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        pts = [Point(float(LonW[i, j]), float(Lat[i, j])) for j in range(W)]
        mask[i, :] = np.array([1 if poly_prepped.contains(pt) else 0 for pt in pts], dtype=np.uint8)

    # save as NetCDF: variable name 'land_mask', semantics 1=land, 0=ocean
    da = xr.DataArray(
        mask,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="land_mask",
    )
    da.to_dataset(name="land_mask").to_netcdf(out_path)
    print("[MAKE_LMASK] wrote", out_path, "shape=", mask.shape, "land_frac=", float(mask.mean()))

    ds.close()


if __name__ == "__main__":
    main()
