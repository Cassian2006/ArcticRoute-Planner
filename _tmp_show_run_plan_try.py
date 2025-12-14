from pathlib import Path
lines = Path('ArcticRoute/api/cli.py').read_text(encoding='utf-8').splitlines()
anchor = next(i for i, line in enumerate(lines) if 'run_report_{tag_value}.json' in line)
block_start = next(i for i in range(anchor, len(lines)) if lines[i].strip() == 'try:')
for i in range(block_start, block_start + 5):
    print(i+1, repr(lines[i]), len(lines[i]) - len(lines[i].lstrip(' ')))
