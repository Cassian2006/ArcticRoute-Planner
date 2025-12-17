import re
from pathlib import Path
txt=Path("reports/hotfix_round7.txt").read_text(encoding="utf-8", errors="ignore")
# 查找 ERROR collecting 或 FAILED 行
m=re.search(r"ERROR collecting (\S+)", txt)
if not m:
    m=re.search(r"FAILED\s+(\S+)\s+", txt)
if m:
    print("FIRST_FAIL_NODEID=", m.group(1))
else:
    print("FIRST_FAIL_NODEID= NOT_FOUND")
