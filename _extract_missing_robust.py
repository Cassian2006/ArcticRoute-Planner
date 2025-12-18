#!/usr/bin/env python
import re
from pathlib import Path

def read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    # Try to strip NULs (UTF-16 mishandled) first
    if b"\x00" in raw:
        try:
            return raw.decode("utf-16", errors="ignore")
        except Exception:
            pass
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode(errors="ignore")

p = Path("reports/collect_only.txt")
text = read_text_any(p)
# Normalize NULs if any
text = text.replace("\x00", "")

mods = sorted(set(re.findall(r"No module named '([^']+)'", text)))
print("MISSING_MODULES_COUNT=", len(mods))
Path("reports/missing_modules.txt").write_text("\n".join(mods), encoding="utf-8")
for m in mods[:30]:
    print(m)


