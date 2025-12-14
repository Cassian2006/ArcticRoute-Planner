from pathlib import Path
lines = Path('ArcticRoute/api/cli.py').read_text(encoding='utf-8').splitlines()
indices = [i for i, line in enumerate(lines) if 'run_report_{tag_value}.json' in line]
print(indices)
