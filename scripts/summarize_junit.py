from __future__ import annotations
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

def main():
    junit = Path("reports/junit.xml")
    if not junit.exists():
        raise SystemExit("missing reports/junit.xml (run pytest --junitxml=reports/junit.xml first)")

    root = ET.parse(junit).getroot()

    # junit may have <testsuites> or <testsuite>
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

    failures = []
    for s in suites:
        for tc in s.findall("testcase"):
            # failure or error
            fail = tc.find("failure")
            err = tc.find("error")
            node = fail if fail is not None else err
            if node is None:
                continue

            classname = tc.attrib.get("classname", "")
            name = tc.attrib.get("name", "")
            msg = (node.attrib.get("message", "") or "").strip()
            text = (node.text or "").strip()

            # exception type heuristic
            first_line = (text.splitlines()[0] if text else msg)
            exc_type = first_line.split(":", 1)[0].strip() if first_line else "UnknownError"

            failures.append({
                "nodeid": f"{classname}::{name}" if classname else name,
                "exc_type": exc_type or "UnknownError",
                "message": msg[:200],
            })

    by_type = Counter(f["exc_type"] for f in failures)
    by_file = Counter((f["nodeid"].split("::")[0] if f["nodeid"] else "") for f in failures)

    out_md = Path("reports/pytest_failures.md")
    out_md.write_text(_render_md(failures, by_type, by_file), encoding="utf-8")

    print("[OK] wrote", out_md)

def _render_md(failures, by_type, by_file):
    lines = []
    lines.append("# Pytest Failures Summary\n")
    lines.append(f"- total failures: **{len(failures)}**\n")

    lines.append("## By exception type (top 20)\n")
    for k, v in by_type.most_common(20):
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## By file (top 30)\n")
    for k, v in by_file.most_common(30):
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Failure list\n")
    for f in failures:
        lines.append(f"- `{f['nodeid']}` — **{f['exc_type']}** — {f['message']}")
    lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    main()
