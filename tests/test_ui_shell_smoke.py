from __future__ import annotations


def test_shell_skin_import() -> None:
    import arcticroute.ui.shell_skin as shell_skin

    assert hasattr(shell_skin, "inject_shell_css")


def test_inject_shell_css_calls_markdown(monkeypatch) -> None:
    import arcticroute.ui.shell_skin as shell_skin

    called = {}

    def fake_markdown(*args, **kwargs) -> None:
        called["called"] = True
        called["kwargs"] = kwargs

    monkeypatch.setattr(shell_skin.st, "markdown", fake_markdown)
    shell_skin.inject_shell_css()

    assert called.get("called") is True
    assert called["kwargs"].get("unsafe_allow_html") is True


def test_planner_minimal_import() -> None:
    import arcticroute.ui.planner_minimal as planner_minimal

    assert hasattr(planner_minimal, "render")
