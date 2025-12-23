import run_ui  # noqa: F401
from arcticroute.core.planners.selector import select_planner_backend


def test_selector_meta_contains_keys():
    _, meta = select_planner_backend("auto")
    for key in ["planner_used", "planner_mode", "fallback_reason"]:
        assert key in meta

