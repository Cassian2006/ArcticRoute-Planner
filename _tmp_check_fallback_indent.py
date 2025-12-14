from pathlib import Path
lines = Path('ArcticRoute/api/cli.py').read_text(encoding='utf-8').splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith('def _build_fallback_execution('))
end = next(i for i in range(start, len(lines)) if lines[i].lstrip().startswith('def ensure_utf8'))
for i in range(start, end):
    indent = len(lines[i]) - len(lines[i].lstrip(' '))
    print(i+1, indent, repr(lines[i]))
