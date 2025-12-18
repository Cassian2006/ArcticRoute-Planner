from __future__ import annotations
import sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parents[1]

    offenders = []
    if (root / "ArcticRoute").exists():
        offenders.append("ArcticRoute/ (uppercase package dir)")
    if (root / "arcticroute.py").exists():
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







