from arcticroute.core.planners.selector import select_planner_backend


def test_pipeline_mode_without_dir_fallbacks_to_astar(tmp_path):
    backend, meta = select_planner_backend("polarroute_pipeline", pipeline_dir=None)
    assert meta["planner_used"] == "astar"
    assert meta["fallback_reason"]
    assert backend.name == "astar"


def test_astar_mode_keeps_astar():
    backend, meta = select_planner_backend("astar")
    assert meta["planner_used"] == "astar"
    assert meta["fallback_reason"] is None
    assert backend.name == "astar"


def test_auto_without_polarroute_fallbacks():
    backend, meta = select_planner_backend("auto")
    assert meta["planner_used"] == "astar"
    assert meta["fallback_reason"]
    assert backend.name == "astar"

