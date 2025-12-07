from pathlib import Path
lines = Path('ArcticRoute/api/cli.py').read_text(encoding='utf-8').splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith('def load_yaml_file'))
for i in range(start, start + 10):
    indent = len(lines[i]) - len(lines[i].lstrip(' '))
    print(i+1, indent, repr(lines[i]))
