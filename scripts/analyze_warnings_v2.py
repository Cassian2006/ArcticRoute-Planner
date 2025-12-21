import re
from collections import Counter

p = "reports/phase13_warnings_full_v2.txt"
with open(p, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

lines = content.splitlines()

# 查找警告行
warn_type = Counter()
by_module = Counter()
by_pair = Counter()

# 匹配格式如: "site-packages/sklearn/metrics/_ranking.py:424: UndefinedMetricWarning:"
mod_re = re.compile(r"(.*?):(\d+):\s*([A-Za-z]+Warning):")
# 匹配格式如: "<frozen importlib._bootstrap>:241: RuntimeWarning:"
simple_re = re.compile(r"([^:]+):(\d+):\s*([A-Za-z]+Warning):")

for line in lines:
    line = line.strip()
    if not line or line.startswith("=") or line.startswith("--"):
        continue
    
    # 尝试匹配两种格式
    m = mod_re.search(line)
    if not m:
        m = simple_re.search(line)
    
    if m:
        file_path = m.group(1).strip()
        wtype = m.group(3)
        
        warn_type[wtype] += 1
        
        # 分类模块
        if "site-packages" in file_path:
            bucket = "site-packages"
        elif "arcticroute" in file_path:
            bucket = "arcticroute"
        elif "tests/" in file_path:
            bucket = "tests"
        else:
            bucket = "other"
            
        by_module[bucket] += 1
        by_pair[(bucket, wtype)] += 1

print("=== Warning count by source bucket ===")
for k, v in by_module.most_common():
    print(f"{k}: {v}")

print("\n=== Top (bucket, type) ===")
for (b, t), v in Counter(by_pair).most_common(15):
    print(f"{b:12s} {t:25s} {v}")

print("\n=== Top warning types ===")
for k, v in warn_type.most_common(15):
    print(f"{k:30s} {v}")

# 查找是否有我们自己的警告
print("\n=== ArcticRoute warnings (if any) ===")
arcticroute_warnings = [(bucket, wtype) for (bucket, wtype) in by_pair.keys() if bucket == "arcticroute"]
if arcticroute_warnings:
    for (bucket, wtype), count in Counter(by_pair).items():
        if bucket == "arcticroute":
            print(f"ARCTICROUTE {wtype}: {count}")
else:
    print("No warnings from arcticroute code - good!")
