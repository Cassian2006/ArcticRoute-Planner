from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
import sys


def test_planner_minimal_smoke(monkeypatch):
    dummy_io = ModuleType("arcticroute.io")
    dummy_io.__path__ = []
    sys.modules["arcticroute.io"] = dummy_io

    dummy_geo = ModuleType("arcticroute.io.geojson_light")
    dummy_geo.read_geojson_lines = lambda *args, **kwargs: []
    dummy_geo.read_geojson_points = lambda *args, **kwargs: []
    sys.modules["arcticroute.io.geojson_light"] = dummy_geo

    dummy_static = ModuleType("arcticroute.io.static_assets")
    dummy_static.get_static_asset_path = lambda *args, **kwargs: None
    sys.modules["arcticroute.io.static_assets"] = dummy_static

    dummy_cost = ModuleType("arcticroute.core.cost")

    @dataclass
    class DummyCostField:
        grid = None
        cost = None
        land_mask = None
        components = None
        edl_uncertainty = None
        meta = None

    dummy_cost.CostField = DummyCostField
    dummy_cost.build_demo_cost = lambda *args, **kwargs: None
    dummy_cost.build_cost_from_real_env = lambda *args, **kwargs: None
    dummy_cost.list_available_ais_density_files = lambda *args, **kwargs: {}
    dummy_cost.discover_ais_density_candidates = lambda *args, **kwargs: []
    dummy_cost.compute_grid_signature = lambda *args, **kwargs: "demo"
    sys.modules["arcticroute.core.cost"] = dummy_cost

    import arcticroute.ui.planner_minimal as planner

    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    @dataclass
    class DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def container(self):
            return self

        def empty(self):
            return self

    class DummyStreamlit(DummyCtx):
        def __init__(self):
            self.session_state = SessionState()
            self.sidebar = self

        def set_page_config(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def caption(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def success(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def header(self, *args, **kwargs):
            return None

        def subheader(self, *args, **kwargs):
            return None

        def write(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def json(self, *args, **kwargs):
            return None

        def metric(self, *args, **kwargs):
            return None

        def dataframe(self, *args, **kwargs):
            return None

        def bar_chart(self, *args, **kwargs):
            return None

        def line_chart(self, *args, **kwargs):
            return None

        def altair_chart(self, *args, **kwargs):
            return None

        def download_button(self, *args, **kwargs):
            return None

        def tabs(self, labels):
            return [self for _ in labels]

        def columns(self, n):
            return [self for _ in range(n)]

        def expander(self, *args, **kwargs):
            return self

        def spinner(self, *args, **kwargs):
            return self

        def selectbox(self, label, options, index=0, **kwargs):
            return options[index] if options else None

        def radio(self, label, options, index=0, **kwargs):
            return options[index] if options else None

        def slider(self, label, min_value=None, max_value=None, value=None, **kwargs):
            return value

        def number_input(self, label, min_value=None, max_value=None, value=None, **kwargs):
            return value

        def text_input(self, label, value="", **kwargs):
            return value

        def checkbox(self, label, value=False, **kwargs):
            return value

        def button(self, label, **kwargs):
            return False

        def rerun(self):
            return None

    dummy_st = DummyStreamlit()
    monkeypatch.setattr(planner, "st", dummy_st, raising=True)

    monkeypatch.setattr(planner, "_summarize_static_assets", lambda: {
        "missing_required": [],
        "missing_optional": [],
    })
    monkeypatch.setattr(planner, "_read_cmems_status", lambda: ({"timestamp": "fake"}, {"fallback_reason": ""}))
    monkeypatch.setattr(planner, "_preview_ports_corridors", lambda: {
        "ports_count": 0,
        "ports_preview": [],
        "corridors_count": 0,
        "corridors_preview": [],
    })
    monkeypatch.setattr(planner, "_load_rules_config", lambda *_args, **_kwargs: {"rules": []})

    planner.render()
