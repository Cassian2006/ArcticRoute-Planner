from __future__ import annotations

import os
from ArcticRoute.io.ais_norm import normalize_record


def make_keymap():
    # 映射常见变体到规范键
    return {
        "MMSI": "mmsi",
        "Timestamp": "ts",
        "Latitude": "lat",
        "Longitude": "lon",
        "SOG": "sog",
        "COG": "cog",
        "Heading": "heading",
        "ShipType": "vessel_type",
        "Length": "loa",
        "Breadth": "beam",
        "NavStatus": "nav_status",
    }


def test_normalize_epoch_seconds():
    raw = {
        "MMSI": 412345678,
        "Timestamp": 1700000000,  # epoch_s
        "Latitude": 45.0,
        "Longitude": 120.0,
        "SOG": 12.3,
    }
    out = normalize_record(raw, make_keymap())
    assert out is not None
    assert out["mmsi"] == 412345678
    assert out["ts"] == 1700000000
    assert out["lat"] == 45.0
    assert out["lon"] == 120.0
    assert out["sog"] == 12.3


def test_normalize_epoch_millis():
    raw = {
        "MMSI": "412345678",
        "Timestamp": 1700000000123,  # epoch_ms
        "Latitude": "30.5",
        "Longitude": "-10.25",
    }
    out = normalize_record(raw, make_keymap())
    assert out is not None
    assert out["ts"] == 1700000000
    assert out["lat"] == 30.5
    assert out["lon"] == -10.25


def test_normalize_iso8601_and_case_insensitive():
    raw = {
        "mmsi": 123456789,  # 直接小写命中
        "Timestamp": "2024-01-02T03:04:05Z",
        "LAT": 10,
        "LON": 20,
        "Breadth": 32,
    }
    out = normalize_record(raw, make_keymap())
    assert out is not None
    assert out["mmsi"] == 123456789
    assert out["ts"] == 1704164645  # 2024-01-02T03:04:05Z -> epoch
    assert out["lat"] == 10
    assert out["lon"] == 20
    assert out["beam"] == 32


def test_invalid_lat_lon_returns_none():
    raw = {
        "MMSI": 1,
        "Timestamp": 1700000000,
        "Latitude": 95.0,  # invalid
        "Longitude": 10.0,
    }
    assert normalize_record(raw, make_keymap()) is None


def test_missing_required_returns_none():
    raw = {"MMSI": 1, "Latitude": 0, "Longitude": 0}
    assert normalize_record(raw, make_keymap()) is None

