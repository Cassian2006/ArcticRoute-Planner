"""
直接测试 SIC 文件加载。
"""

from pathlib import Path
import xarray as xr

sic_path = Path(r"C:\Users\sgddsf\Desktop\AR_final\data_real\202412\sic_202412.nc")

print(f"SIC file: {sic_path}")
print(f"Exists: {sic_path.exists()}")

if sic_path.exists():
    try:
        ds = xr.open_dataset(sic_path, decode_times=False)
        print(f"\n✓ Successfully opened SIC file")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        print(f"  Coordinates: {list(ds.coords.keys())}")
        
        # Check SIC variable
        if 'sic' in ds:
            sic = ds['sic']
            print(f"\n  SIC variable:")
            print(f"    Shape: {sic.shape}")
            print(f"    Dims: {sic.dims}")
            print(f"    Data type: {sic.dtype}")
        
        ds.close()
    except Exception as e:
        print(f"\n✗ Failed to open SIC file: {e}")
else:
    print(f"✗ SIC file does not exist")

