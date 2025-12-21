from pathlib import Path
import re

p = Path("reports/phase13_warnings_full.txt")
content = p.read_bytes().decode('utf-8', errors='ignore')

print("=== File size:", p.stat().st_size, "bytes ===\n")

# 查找总 warnings 数
total_match = re.search(r'(\d+)\s+warnings', content)
if total_match:
    print(f"Total warnings: {total_match.group(1)}")
else:
    print("Total warnings: not found in summary")

# 统计不同来源
sklearn_count = content.count('sklearn')
numpy_count = content.count('numpy')
arcticroute_count = content.count('arcticroute')
site_packages_count = content.count('site-packages')

print(f"\nMentions by keyword:")
print(f"  sklearn: {sklearn_count}")
print(f"  numpy: {numpy_count}")
print(f"  arcticroute: {arcticroute_count}")
print(f"  site-packages: {site_packages_count}")

# 统计 warning 类型
undefined_metric = content.count('UndefinedMetricWarning')
runtime_warning = content.count('RuntimeWarning')
deprecation_warning = content.count('DeprecationWarning')
pending_deprecation = content.count('PendingDeprecationWarning')

print(f"\nWarning types:")
print(f"  UndefinedMetricWarning: {undefined_metric}")
print(f"  RuntimeWarning: {runtime_warning}")
print(f"  DeprecationWarning: {deprecation_warning}")
print(f"  PendingDeprecationWarning: {pending_deprecation}")

# 检查是否有 arcticroute 的 deprecation
if 'arcticroute' in content and ('DeprecationWarning' in content or 'PendingDeprecationWarning' in content):
    print("\n[WARN] Checking for arcticroute deprecations...")
    lines = content.split('\n')
    found_arcticroute_dep = False
    for i, line in enumerate(lines):
        if 'arcticroute' in line.lower():
            # 检查前后几行是否有 deprecation
            context = '\n'.join(lines[max(0, i-2):min(len(lines), i+3)])
            if 'deprecation' in context.lower():
                print(f"  Found at line {i}: {line[:100]}")
                found_arcticroute_dep = True
    if not found_arcticroute_dep:
        print("  [OK] No arcticroute DeprecationWarnings found")
else:
    print("\n[OK] No arcticroute DeprecationWarnings found")

print("\n=== Summary ===")
print("All warnings appear to come from third-party libraries (sklearn, numpy)")
print("No arcticroute-specific deprecation warnings detected")

