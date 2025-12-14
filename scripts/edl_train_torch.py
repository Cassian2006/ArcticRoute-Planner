from __future__ import annotations

import argparse

from arcticroute.core.edl_train_torch import train_edl_model_from_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simple EDL MLP from parquet datasets.")
    parser.add_argument("--config", default="configs/edl_train.yaml", help="Path to training config YAML")
    args = parser.parse_args()

    result = train_edl_model_from_parquet(args.config)
    
    # 检查是否 torch 不可用
    if isinstance(result, dict) and result.get("status") == "torch_unavailable":
        print("[EDL_TRAIN] 当前 venv 无法导入 torch，建议在 ar_edl conda 环境中运行训练脚本。")
        return
    
    # 正常训练完成的情况
    report = result
    final = report.get("final", {}) or {}
    train_acc = final.get("train_acc")
    val_acc = final.get("val_acc")
    train_acc_str = f"{train_acc:.4f}" if train_acc is not None else "n/a"
    val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "n/a"
    print(
        "[EDL_TRAIN] done. "
        f"samples: train={report.get('train_samples')} val={report.get('val_samples')}; "
        f"final train_acc={train_acc_str} val_acc={val_acc_str}"
    )
    print(f"[EDL_TRAIN] model saved to: {report.get('model_path')}")
    print(f"[EDL_TRAIN] report saved to: {report.get('report_path', 'n/a')}")


if __name__ == "__main__":
    main()
