from pathlib import Path
lines = Path('ArcticRoute/api/cli.py').read_text(encoding='utf-8').splitlines()
block_start = next(i for i, line in enumerate(lines) if line.strip() == 'try:')
except1_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('except FileNotFoundError'))
except2_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('except Exception'))
block_end = except2_idx
while block_end < len(lines) and lines[block_end].strip() != '':
    block_end += 1
block_end += 1
print('indices', block_start, except1_idx, except2_idx, block_end)
print('try line repr:', repr(lines[block_start]))
