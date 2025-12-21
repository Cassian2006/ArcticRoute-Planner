import re
from collections import Counter, defaultdict
from pathlib import Path
p = Path("reports/phase13_warnings_full.txt")
if not p.exists():
    print("MISSING: reports/phase13_warnings_full.txt")
    raise SystemExit(1)

txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()

warn_type = Counter()
by_module = Counter()
by_pair = Counter()

mod_re = re.compile(r"^(.*\.py):\d+:\s*([A-Za-z]+Warning):")
type_re = re.compile(r"([A-Za-z]+Warning):")

for line in txt:
    m = mod_re.search(line.strip())
    if m:
        mod_path, wtype = m.group(1), m.group(2)
        warn_type[wtype] += 1
        mod = mod_path.replace("\\", "/")
        # coarse bucket
        if "/site-packages/" in mod or "site-packages" in mod:
            bucket = "site-packages"
        elif mod.startswith("arcticroute/") or "/arcticroute/" in mod:
            bucket = "arcticroute"
        elif mod.startswith("tests/") or "/tests/" in mod:
            bucket = "tests"
        else:
            bucket = "other"
        by_module[bucket] += 1
        by_pair[(bucket, wtype)] += 1
    else:
        m2 = type_re.search(line)
        if m2 and "Warning" in m2.group(1):
            warn_type[m2.group(1)] += 0

print("=== Warning count by source bucket ===")
for k,v in by_module.most_common():
    print(f"{k}: {v}")

print("\n=== Top (bucket, type) ===")
for (b,t),v in Counter(by_pair).most_common(15):
    print(f"{b:12s} {t:25s} {v}")

print("\n=== Top warning types ===")
for k,v in warn_type.most_common(15):
    print(f"{k:30s} {v}")

