from pathlib import Path
need = [
    Path("arcticroute/__init__.py"),
    Path("arcticroute/core/__init__.py"),
    Path("arcticroute/core/grid.py"),
]
for p in need:
    print(p, "EXISTS" if p.exists() else "MISSING")

