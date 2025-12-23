import importlib, inspect
import arcticroute
import arcticroute.core.eco as eco
print("eco module file:", eco.__file__)
import pathlib
p = pathlib.Path(eco.__file__)
print("eco __init__.py content:\n---\n" + p.read_text(encoding="utf-8", errors="ignore") + "\n---")







