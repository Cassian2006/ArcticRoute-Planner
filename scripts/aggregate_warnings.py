import re
from collections import Counter
from pathlib import Path

p = Path("reports/phase13_warnings_full.txt")
if not p.exists():
    print("WARN: reports/phase13_warnings_full.txt not found")
    raise SystemExit(1)

txt = p.read_text(encoding="utf-8", errors="ignore")

# 提取 warnings 统计
warn_type = Counter()
by_module = Counter()
by_pair = Counter()

# 匹配格式: "  C:\...\file.py:line: WarningType: message"
# 或 "  <frozen importlib._bootstrap>:241: RuntimeWarning: ..."
pattern = re.compile(r'^\s+(.+?):\d+:\s*([A-Za-z]+Warning):', re.MULTILINE)

for match in pattern.finditer(txt):
    mod_path, wtype = match.group(1), match.group(2)
    warn_type[wtype] += 1
    
    mod = mod_path.replace("\\", "/")
    
    # 分类
    if "site-packages" in mod or "/site-packages/" in mod:
        bucket = "site-packages"
    elif "<frozen" in mod or "importlib" in mod:
        bucket = "stdlib"
    elif "arcticroute/" in mod or mod.startswith("arcticroute"):
        bucket = "arcticroute"
    elif "tests/" in mod or mod.startswith("tests"):
        bucket = "tests"
    else:
        bucket = "other"
    
    by_module[bucket] += 1
    by_pair[(bucket, wtype)] += 1

# 提取总数
total_match = re.search(r'(\d+)\s+warnings', txt)
total_warnings = int(total_match.group(1)) if total_match else 0

print(f"=== Total warnings: {total_warnings} ===\n")

print("=== Warning count by source bucket ===")
if by_module:
    for k, v in by_module.most_common():
        print(f"{k}: {v}")
else:
    print("(no warnings with parseable source)")

print("\n=== Top (bucket, type) ===")
if by_pair:
    for (b, t), v in Counter(by_pair).most_common(15):
        print(f"{b:12s} {t:25s} {v}")
else:
    print("(no warnings with parseable bucket+type)")

print("\n=== Top warning types ===")
if warn_type:
    for k, v in warn_type.most_common(15):
        print(f"{k:30s} {v}")
else:
    print("(no warnings with parseable type)")

# 检查是否有来自 arcticroute 的 DeprecationWarning
arcticroute_deprecations = [(b, t, v) for (b, t), v in by_pair.items() 
                            if b == "arcticroute" and "Deprecation" in t]
if arcticroute_deprecations:
    print("\n⚠️  WARNING: Found arcticroute DeprecationWarnings!")
    for b, t, v in arcticroute_deprecations:
        print(f"  {b} {t}: {v}")
else:
    print("\n✅ No arcticroute DeprecationWarnings found")
