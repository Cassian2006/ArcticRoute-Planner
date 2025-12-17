from pathlib import Path
from arcticroute.core.ais_ingest import load_ais_from_raw_dir
import tempfile
import shutil

# 复制测试数据
src_json = Path("tests/data/ais_sample.json")
src_csv = Path("tests/data/ais_sample.csv")

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    dst_json = tmp_path / "ais_sample.json"
    dst_csv = tmp_path / "ais_sample.csv"
    
    dst_json.write_text(src_json.read_text(), encoding="utf-8")
    dst_csv.write_text(src_csv.read_text(), encoding="utf-8")
    
    df = load_ais_from_raw_dir(tmp_path, prefer_json=True)
    
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("\nData:")
    print(df)
    print("\nLat values:")
    print(df["lat"].tolist())
    print("\nLon values:")
    print(df["lon"].tolist())

