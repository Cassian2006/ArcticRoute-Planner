from __future__ import annotations


def test_shell_css_extracts_variables() -> None:
    from arcticroute.ui.shell_skin import extract_shell_css

    css = extract_shell_css()
    assert css.strip()
    assert ":root" in css or "--bg" in css


def test_planner_minimal_imports() -> None:
    import arcticroute.ui.planner_minimal  # noqa: F401
