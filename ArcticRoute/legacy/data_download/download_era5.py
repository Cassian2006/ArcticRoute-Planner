import cdsapi
from pathlib import Path

# 说明：
# 1) 先在 ~/.cdsapirc 写入你的 CDS API key
# 2) 可修改 year/month/day/time/area
# 3) 输出到 data_raw/era5/


def main():
    out_dir = Path(__file__).resolve().parents[1] / "data_raw" / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    target = out_dir / "era5_env_2023_q1.nc"
    print(f"Downloading to {target} ...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "significant_height_of_combined_wind_waves_and_swell",
                "sea_ice_cover",
            ],
            "year": ["2023"],
            "month": ["01", "02", "03"],
            "day": ["01", "05", "10", "15", "20", "25"],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [85, -180, 60, 180],  # N, W, S, E
            "format": "netcdf",
        },
        str(target),
    )
    print("Done.")


if __name__ == "__main__":
    main()
