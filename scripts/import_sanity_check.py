from __future__ import annotations
import sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parents[1]

    offenders = []
    # On case-insensitive filesystems (e.g., Windows NTFS),
    # Path("ArcticRoute").exists() may return True even if the actual directory
    # is named "arcticroute" (lowercase). To avoid false positives, enumerate
    # actual entry names and compare case-sensitively.
    try:
        entries = {p.name for p in root.iterdir()}
    except Exception:
        entries = set()
    if "ArcticRoute" in entries:
        offenders.append("ArcticRoute/ (uppercase package dir)")
    if "arcticroute.py" in entries:
        offenders.append("arcticroute.py (shadowing package)")

    if offenders:
        msg = (
            "IMPORT SANITY CHECK FAILED\\n"
            "These files/dirs can shadow the 'arcticroute' package and cause import errors on Windows:\\n"
            + "\\n".join(f"- {x}" for x in offenders)
            + "\\n\\nFix (local only):\\n"
            "  - delete ArcticRoute/ directory (if it is a legacy copy)\\n"
            "  - delete arcticroute.py (shadow module)\\n"
            "  - delete __pycache__/ then re-run tests\\n"
        )
        print(msg, file=sys.stderr)
        return 2

    # also ensure we import the correct package
    import arcticroute  # noqa: F401
    print("IMPORT SANITY CHECK PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())







