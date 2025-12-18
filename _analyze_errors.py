#!/usr/bin/env python
import re
from pathlib import Path

txt = Path("reports/collect_only_after_restore.txt").read_text(encoding="utf-8", errors="ignore")

# 提取所有 ERROR collecting 行
errors = re.findall(r"ERROR collecting (tests/[^\s]+)", txt)
print(f"ERROR collecting count = {len(errors)}")
for e in errors:
    print(f"  {e}")

# 提取 ModuleNotFoundError 和 cannot import name
mod_errors = re.findall(r"ModuleNotFoundError: No module named '([^']+)'", txt)
import_errors = re.findall(r"cannot import name '([^']+)' from '([^']+)'", txt)

print(f"\nMissing modules: {len(set(mod_errors))}")
for m in sorted(set(mod_errors)):
    print(f"  {m}")

print(f"\nMissing symbols: {len(set(import_errors))}")
for name, mod in sorted(set(import_errors)):
    print(f"  {mod}::{name}")



