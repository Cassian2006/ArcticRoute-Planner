from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


def load_dataset(csv_path: Optional[Path], target_col: str):
    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"target column '{target_col}' not found in CSV")
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).astype(float)
        y = df[target_col].astype(float)
        if X.empty:
            raise ValueError("no numeric features found after dropping target")
        return X, y
    # 合成一个最小演示数据集：
    # 特征：segment_length_nm, wind_speed, wave_height, ice_risk(0..1), vessel_class_ix(0/1)
    # 目标：fuel_per_nm（吨/海里）
    rng = np.random.default_rng(42)
    n = 2000
    seg_len = rng.uniform(1.0, 30.0, size=n)
    wind = rng.uniform(0.0, 20.0, size=n)
    wave = rng.uniform(0.0, 6.0, size=n)
    ice = rng.uniform(0.0, 1.0, size=n)
    vclass_ix = rng.integers(0, 2, size=n)  # 0=cargo_standard, 1=cargo_iceclass
    base = np.where(vclass_ix == 1, 0.015, 0.012)
    # 规则：风/浪/冰 增加 per-nm，加入少量噪声
    per_nm = base * (1.0 + 0.02 * wind + 0.06 * wave + 0.15 * ice) * rng.normal(1.0, 0.03, size=n)
    df = pd.DataFrame({
        "segment_length_nm": seg_len,
        "wind_speed": wind,
        "wave_height": wave,
        "ice_risk": ice,
        "vclass_ix": vclass_ix,
        target_col: per_nm,
    })
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def main():
    ap = argparse.ArgumentParser(description="Train a minimal sklearn regressor and export as .joblib")
    ap.add_argument("--csv", default=None, help="Optional CSV path; if omitted, synthetic data is used")
    ap.add_argument("--target", default="fuel_per_nm", help="Target column name in CSV (default fuel_per_nm)")
    ap.add_argument("--out", default=str(Path("C:/models/fuel_model.joblib")), help="Output .joblib path")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(csv_path, args.target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)

    model = GradientBoostingRegressor(random_state=7)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    r2 = float(r2_score(y_te, y_pred))
    mae = float(mean_absolute_error(y_te, y_pred))

    joblib.dump({"model": model, "feature_names": list(X.columns), "target": args.target}, out_path)
    print(json.dumps({"ok": True, "out": str(out_path), "n_features": X.shape[1], "r2": r2, "mae": mae}))


if __name__ == "__main__":
    main()



