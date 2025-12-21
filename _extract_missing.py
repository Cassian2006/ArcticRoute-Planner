#!/usr/bin/env python
import re
from pathlib import Path

# 提取缺模块
txt = Path("reports/collect_only.txt").read_text(encoding="utf-8", errors="ignore")
mods = sorted(set(re.findall(r"No module named '([^']+)'", txt)))
print(f"MISSING_MODULES_COUNT= {len(mods)}")
Path("reports/missing_modules.txt").write_text("\n".join(mods), encoding="utf-8")
for m in mods[:30]:
    print(m)

# 提取缺符号
pat = r"cannot import name '([^']+)' from '([^']+)'"
hits = sorted(set(re.findall(pat, txt)))
print(f"\nMISSING_SYMBOLS_COUNT= {len(hits)}")
Path("reports/missing_symbols.txt").write_text("\n".join([f"{mod}::{name}" for name, mod in hits]), encoding="utf-8")
for name, mod in hits[:30]:
    print(f"{mod} -> {name}")




