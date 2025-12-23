from __future__ import annotations
import json
from pathlib import Path
from arcticroute.ui.ui_doctor import run_ui_doctor

def main() -> int:
    checks = run_ui_doctor()
    out = [{"id":c.id,"level":c.level,"title":c.title,"detail":c.detail,"fix":c.fix} for c in checks]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/ui_doctor.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    fail = [c for c in checks if c.level == "FAIL"]
    print("[ui-doctor] report -> reports/ui_doctor.json")
    for c in checks:
        print(f"{c.level:4} {c.id:14} {c.title}: {c.detail}")
    return 2 if fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
