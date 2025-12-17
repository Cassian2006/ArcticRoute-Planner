from pathlib import Path
txt=Path("reports/hotfix_round7.txt").read_text(encoding="utf-8", errors="ignore")
lines = txt.split('\n')
print('\n'.join(lines[-30:]))

