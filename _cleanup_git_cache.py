#!/usr/bin/env python3
"""清理 Git 缓存中的大文件"""
import subprocess
import sys
from pathlib import Path

def run_git_rm(pattern):
    """运行 git rm --cached"""
    try:
        result = subprocess.run(
            ["git", "rm", "-r", "--cached", pattern],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        if result.returncode == 0:
            print(f"[OK] Untracked: {pattern}")
            return True
        elif "did not match any files" in result.stderr:
            print(f"[SKIP] No matching files: {pattern}")
            return True
        else:
            print(f"[FAIL] {pattern}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def main():
    patterns = [
        "data/cmems_cache",
        "ArcticRoute/data_processed/newenv",
        "reports/cmems_sic_describe.json",
        "reports/cmems_swh_describe.json",
        "reports/cmems_wav_describe.json",
        "reports/cmems_*_probe.json",
        "reports/cmems_refresh_last.json",
        "reports/cmems_resolved.json",
    ]
    
    print("Starting cleanup of large files in Git cache...")
    print("=" * 60)
    
    all_ok = True
    for pattern in patterns:
        if not run_git_rm(pattern):
            all_ok = False
    
    print("=" * 60)
    if all_ok:
        print("[DONE] Cleanup completed")
        return 0
    else:
        print("[WARN] Some cleanup failed, but continuing")
        return 0

if __name__ == "__main__":
    sys.exit(main())

