from pathlib import Path
lines = Path('ArcticRoute/api/cli.py').read_text(encoding='utf-8').splitlines()
start_func = next(i for i, line in enumerate(lines) if line.startswith('def run_plan('))
block_start = next(i for i in range(start_func, len(lines)) if lines[i].strip() == 'try:')
except1_idx = next(i for i in range(block_start, len(lines)) if lines[i].strip().startswith('except FileNotFoundError'))
except2_idx = next(i for i in range(except1_idx + 1, len(lines)) if lines[i].strip().startswith('except Exception'))
block_end = except2_idx
while block_end < len(lines) and lines[block_end].strip() != '':
    block_end += 1
block_end += 1
print('indices', start_func, block_start, except1_idx, except2_idx, block_end)
print('try repr', repr(lines[block_start]))
print('except1 repr', repr(lines[except1_idx]))
print('except2 repr', repr(lines[except2_idx]))
